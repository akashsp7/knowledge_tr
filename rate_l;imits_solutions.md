# MY QUERY

Hi! I am making a project and running into a lot of problems. I am going to define what the project is plus the problems I am having.

Project: A morning note creator, which searches for news across our coverage universe of tickers every 24 hours and creates a morning note and posts it on slack.

Goal: The note should have all the news which are important enough to be seen by a PM. These should not be analysts fluffs or ups and downs in stock but rather concrete changes in fundamentals, new product launches etc. Our job is to predict what others don't know yet, so we want to find signals which would help us predict that.

This project, I have currently hosted a stable version on ec2 linux instance which works nice. Then i made additional modifications after which code is crashing.

Let me go into technicalities to shed some more light (the reason I cannot share the code, is because its on work computer and I cannot bring it out) so I will try my best to explain the process here.

first, we had 3 llm steps with a small dedupe step between 1st and 2nd step.

First step: We fetch news. Our coverage universe is around 273 tickers, and we have 27 groups (for eg: csps_and_datacenters, private_ai_companies, ai_semis, etc..). For each group, we create 1 worker (max_workers = 27) in a thread pool executor. Each worker is a single llm call, and we give it a nice prompt to fetch news relevant to what we want. We use gpt-5.1 with reasoning set to high and verbosity set to high. We use the responses api, with response.parse method.

Dedupe: Chromadb has news stored from previous runs, quick similarity check and dropping here.

Second Step: Each groups raw bullets are summarized (group summary is created basically).

Third Step: All news are fed to the llm (Curated final ones) and it chooses top 10 for us. These appear at the top.

Now, this is hosted on the VM and it works nice, everyday, although the quality and dedupe needed improvements. For this I made additional changes but not I am having problems, mostly with OpenAI timeouts. Please note that the above 27 max workers have no token limit or time limit set but they still work.

Now the current implementation and its problems.

First we have RSS fetching, simple script to rss some sites we setup and store it in a light sqlite db.
These only extract the title, article link, date, and a short summary if available

Then we send these to llm (gpt-5.1 with reasoning set to 'none'). Pretty quick calls but alot of calls happen here with a thread pool executor. we basically ask it to keep or drop each rss headline as per our TMT-relevant rules (our coverage universe is TMT sector).

After this, we have 1 llm call per group, and we feed this all the news (chunked actually, so if we have 100 news, and chunk size 25, we make 4 calls for the same group). This step is light, where we group news into relevant group (like csps_and_datacenters etc etc). so it receives the names of companies in the group and the news, and decides which news belong, thats it.

Then we summarize these rss items (lol, just had an idea typing this, that we don't cap how many bullets per group we are sending). We do chunk the groups however (for example, only 3 companies per group in one call) so the same group goes multiple times. I know as i type these that things could be modified here, but the problem is occurring later. Also I plan to add this step to dig deeper if the news in important but thin from the rss fetch. It can either do this using websearch or later, i will add article fetcher to give it access to paid articles.

Then, we have the web fetch step which is the heaviest (first step in the vm implementation). This step is failing constantly in this implementation by the way.

Then, there is improved dedupe, where we do similarity and then above a certain threshold, send it to a light llm for comparison (same, same with additional info, different). The additional info and different tagged news are kept

The we merge the rss and web fetch news bundle and do the downstream steps (summarization per group and top news).

My questions are: How does the same step which worked fine before is timing out now (the timeouts, are worse by the way, even when i set a timeout of 600s, it sometimes goes to 30-60 minutes.) I tried lowering maxworkers to 16 too.

One thing i cannot compromise on is the high reasoning for fetch and rss bullet steps, since we need that level for quality outputs.

Another question is possible improvements in architecture, how to deal with OpenAI timeouts in the most optimal and robust way (we want to manage latency but first priority is quality). Right now the VM version is done within 30 minutes, we can stretch this to an hour or 2 max for the improved version.

# RESPONSE

What you’re seeing (a “600s timeout” turning into **30–60 minutes**) is very consistent with **(1) retry behavior** + **(2) you massively increased request volume before the heavy step**, so you’re much more likely to hit throttling/backpressure.

### Why it “worked before” but times out now

1. **Your pipeline isn’t equivalent anymore (request count exploded).**
   Old version: ~27 big calls for “fetch news per group”, then a couple downstream steps.
   New version: _many_ RSS “keep/drop” calls + chunked routing + chunked summarization + web fetch. Even if those early calls are “reasoning:none”, they still consume **RPM/TPM budget** and can push you into rate limiting / congestion, making the later heavy web-fetch calls slow/fail.

2. **The OpenAI Python SDK can turn one timeout into ~30 minutes by default.**
   The official `openai` Python SDK:

- **Defaults to a 10-minute request timeout** and
- **Retries certain errors 2 times by default**, including timeouts and 429s. ([GitHub][1])

So if an attempt runs ~10 minutes and times out, you can get roughly:

- 10 min (attempt 1) + 10 min (attempt 2) + 10 min (attempt 3) ≈ **30 minutes**
  …and if you also added your own retry loop on top, or configured higher retries somewhere, **60 minutes is totally plausible**.

Also: setting `timeout=600` may not have changed anything, because **10 minutes is already the default**. ([GitHub][1])

3. **If you’re not capping output tokens, you’re “lying” to the rate limiter.**
   Rate limit accounting can be driven by the larger of “max tokens” vs estimated prompt size; setting very high token ceilings can cause earlier throttling than you expect. The OpenAI cookbook explicitly calls this out and recommends setting max tokens close to what you expect. ([OpenAI Cookbook][2])

---

## The most impactful fixes (quality-first, robust)

### A) Make wall-clock timeouts real: control **retries** + **timeouts** explicitly

If you want “never spend more than ~10 minutes on a single call”, you must set `max_retries=0` (or very low) and implement your own policy.

```python
import httpx, time, random
import openai
from openai import OpenAI

client = OpenAI(
    timeout=httpx.Timeout(600.0, connect=10.0, read=600.0, write=30.0),
    max_retries=0,  # IMPORTANT: prevents 30-min+ “timeout cascades”
)

def call_with_backoff(make_call, *, deadline_s=900, max_attempts=3):
    t0 = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return make_call()
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            if attempt >= max_attempts or (time.monotonic() - t0) > deadline_s:
                raise
            # jittered exponential backoff
            sleep_s = min(60, (2 ** (attempt - 1))) + random.random()
            time.sleep(sleep_s)
```

Key point: **SDK timeouts are retried twice by default** unless you change it. ([GitHub][1])
Also enable SDK logging to confirm what’s happening (`OPENAI_LOG=info|debug`). ([GitHub][1])

### B) Add stage-level “concurrency budgets” (not one global max_workers)

You want something like:

- RSS keep/drop: high throughput, **low reasoning**, concurrency maybe 8–16
- Group routing: concurrency maybe 4–8
- Web fetch (high reasoning): concurrency maybe **1–3** (seriously)

A simple pattern is a `Semaphore` per stage so your heavy calls can’t get starved by the firehose.

### C) Batch the RSS keep/drop calls (this is huge)

Per-headline LLM calls are a timeout/rate-limit magnet.

Instead:

- Send **50–200 RSS items per call**
- Return structured JSON: `[{id, keep, reason, tickers?}]`

This cuts calls by 50–200x immediately, and your later high-reasoning steps will stop getting throttled by earlier noise.

### D) Cap what you send into summarization / web fetch

You already noticed it while typing: if a group has 200 bullets and you “summarize” them with high verbosity, you’re guaranteeing long runtimes.

Do this instead:

1. **Score & rank** items first (cheap model or low-reasoning prompt): “fundamental impact / novelty / confidence”
2. Only the **top K per group** (say K=10–25) get:

   - deeper fetch
   - high-reasoning synthesis

This keeps quality high because the model spends thought on the right candidates.

### E) Reduce output length without reducing reasoning

You said you can’t compromise on reasoning for key steps—fine. But you _can_ usually lower latency a lot by forcing concise outputs:

- set `max_output_tokens` to something sane
- require “max N bullets, each <= 25 words”
- use Structured Outputs / strict JSON so it can’t ramble

### F) Make failures non-fatal: checkpoint + partial completion

For a morning note, “90% complete on time” beats “100% never ships”.

Persist per-stage artifacts (you’re already using sqlite + chroma):

- if web fetch for group X fails, proceed with others
- include a tiny “coverage gaps” footer in Slack (internal transparency)
- reruns only redo failed groups, not the whole pipeline

---

## Why your heavy “web fetch step” is failing _specifically_ now

Most likely: it’s not that the model suddenly got worse—it’s that by the time you reach web fetch, you’ve already burned RPM/TPM and triggered throttling. Then:

- requests slow down,
- some timeout,
- SDK retries (default),
- wall time explodes.

This aligns exactly with the cookbook guidance on rate limits + backoff, and with the SDK’s default retry + 10-minute timeout behavior. ([OpenAI Cookbook][2])

---

## A “quality-first but finishes in <2 hours” architecture I’d use

1. **RSS ingest** (no LLM)
2. **Batch filter** (few calls)
3. **Batch tag-to-groups** (few calls, or embeddings + small LLM for uncertain)
4. **Impact scoring** (few calls)
5. **Fetch full text only for top candidates**
6. **High-reasoning synthesis** only on those
7. Dedupe + final top-10 + Slack

This preserves high reasoning where it matters, and makes timeouts rare because you stop doing expensive work on low-signal items.

If you implement only two things: **(1) `max_retries=0` + your own bounded retry** and **(2) batch RSS filtering**, you’ll probably see the biggest immediate improvement.

[1]: https://raw.githubusercontent.com/openai/openai-python/main/README.md "raw.githubusercontent.com"
[2]: https://cookbook.openai.com/examples/how_to_handle_rate_limits "How to handle rate limits"

# MY QUERY

Are we sure there aren't any other problems? My organization is in tier 5 for openai (highest) and both the rpm and tpm are extremely high. Also, can we use the tenacity library for better handling of these limits (employing retry, jitter etc).

Then for max tokens, the reason I am setting it to none is that we cannot predict how much tokens the internal reasoning + web fetches would consume. Also, I don't want a half response.

The bullets are already restricted to be a certain number, i thought that would be cleaner than max_output_tokens. Using parse with pydantic structured outputs.

I understand that those rate limits _might_ be for the whole organization, so yeah, we can chill a bit, but i do want some retries and backoffs (robust system so even if we fail we retry reasonable amount of times)

# RESPONSE

Yeah—**even at Tier 5**, there are several _non-RPM/TPM_ ways you can end up with “600s” turning into **30–60 minutes**, and they’re common in high-concurrency pipelines.

### Other likely culprits (besides raw RPM/TPM)

1. **You’re probably seeing “timeout × retries” (and possibly “retry stack”).**
   In `openai-python`, **requests time out after 10 minutes by default** and **certain errors (including timeouts) are retried 2 times by default**. So a per-attempt timeout of 600s can become ~**30 minutes** (600 × 3), and if you’ve wrapped _another_ retry layer around it, you can easily get to ~60 minutes. ([GitHub][1])
   ✅ Fix: if you use Tenacity, set `max_retries=0` in the OpenAI client to avoid stacked retries. ([GitHub][1])

2. **HTTP connection pool queueing (looks like “hangs”, not rate limits).**
   `httpx` (used by the SDK) has a connection pool; by default it allows **max 100 total connections** and **20 keep-alive**. Extra concurrent requests can **block waiting for a connection**, and that wait time is _outside_ what many people mentally count as “API timeout”. ([python-httpx.org][2])
   ✅ Fix: (a) don’t let early stages flood concurrency, and/or (b) raise httpx pool limits.

3. **Tier 5 doesn’t prevent “burst” limiting or org/project contention.**
   Rate limits can be applied over shorter windows than “per minute,” so big bursts can still trip 429/backpressure behavior. ([OpenAI Help Center][3])
   Also: even if the org is Tier 5, **project-level limits can be set per model** and can be lower than you think. ([OpenAI Help Center][4])

4. **Tool-heavy / “web fetch” style calls can legitimately have long tail latency.**
   Even with high quotas, a reasoning-heavy call that involves browsing/tool use can just be slow. If those calls are also being retried automatically (point #1), it gets ugly fast.

---

## Yes—Tenacity is a good idea (but do it in a way that stays bounded)

Key principles:

- **Disable SDK auto-retries** (`max_retries=0`) so _you_ control policy. ([GitHub][1])
- Retry **429 / 5xx / network / timeouts**, not 4xx “bad request”.
- For 429, prefer respecting retry headers / backoff guidance. ([OpenAI Platform][5])
- Enforce a **deadline** (stop-after-delay) so one call can’t eat 60 minutes.

Here’s a solid Tenacity wrapper pattern:

```python
import logging, time
import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay
from tenacity import wait_random_exponential, before_sleep_log

logger = logging.getLogger("morning-note")

client = OpenAI(
    timeout=600.0,
    max_retries=0,  # IMPORTANT: avoid stacking retries with Tenacity
)

def is_retryable(exc: Exception) -> bool:
    # Network / timeouts
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True

    # API status errors
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500

    return False

def wait_with_retry_after(retry_state):
    exc = retry_state.outcome.exception()
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 429 and exc.response is not None:
        ra = exc.response.headers.get("retry-after")
        if ra:
            try:
                return float(ra)  # usually seconds
            except ValueError:
                pass
    # fallback: jittered exponential backoff
    return wait_random_exponential(min=1, max=60)(retry_state)

@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_with_retry_after,
    stop=(stop_after_attempt(6) | stop_after_delay(900)),  # bounded
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def responses_call(**kwargs):
    resp = client.responses.create(**kwargs)
    # Optional observability
    logger.info("openai_request_id=%s", getattr(resp, "_request_id", None))  # _request_id supported :contentReference[oaicite:7]{index=7}
    return resp
```

---

## About `max_output_tokens`: your reasoning is valid, and there’s a nuance

In the Responses API, `max_output_tokens` is **an upper bound for generated tokens including visible output tokens _and reasoning tokens_**. ([OpenAI Platform][6])
So if you set it too low, you can absolutely force “incomplete” outputs. The reasoning guide explicitly says you’ll get **status = `incomplete`** if you hit the context window limit or your `max_output_tokens` value. ([OpenAI Platform][7])

Given that:

- Leaving it unset is a defensible choice for your “high reasoning + tool” steps (if you truly cannot tolerate truncation).
- The practical compromise is: **set a very high cap per step** (not “none”), log actual usage, and only tighten later once you’ve observed your true p95/p99.

Also, with Structured Outputs: if you ever _do_ get `incomplete`, treat it as a **retry-with-higher-cap** (or rerun with slightly different constraints) rather than trying to parse-and-pray.

---

## The single biggest robustness win (without sacrificing quality)

Add **bulkheads**: per-stage concurrency limits so the RSS filtering flood can’t starve the “web fetch + high reasoning” workers.

Even with Tier 5, this prevents:

- connection pool queueing ([python-httpx.org][2])
- bursty short-window throttles ([OpenAI Help Center][3])
- tail latency amplification from retries ([GitHub][1])

---

If you implement just these three changes, you’ll usually see the whole system stabilize:

1. `max_retries=0` + Tenacity with a hard stop-after-delay ([GitHub][1])
2. stage-level concurrency caps (especially before the heavy web fetch step) ([python-httpx.org][2])
3. log request IDs + error types so you can prove whether it’s queueing, throttling, or genuine latency ([GitHub][1])

[1]: https://github.com/openai/openai-python?utm_source=chatgpt.com "The official Python library for the OpenAI API"
[2]: https://www.python-httpx.org/advanced/resource-limits/?utm_source=chatgpt.com "Resource Limits"
[3]: https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors?utm_source=chatgpt.com "How can I solve 429: 'Too Many Requests' errors?"
[4]: https://help.openai.com/es-es/articles/9186755-managing-projects-in-the-api-platform?utm_source=chatgpt.com "Managing projects in the API platform"
[5]: https://platform.openai.com/docs/guides/error-codes?utm_source=chatgpt.com "Error codes - OpenAI API"
[6]: https://platform.openai.com/docs/api-reference/responses?utm_source=chatgpt.com "API Reference - OpenAI API"
[7]: https://platform.openai.com/docs/guides/reasoning?utm_source=chatgpt.com "Reasoning models - OpenAI API"
