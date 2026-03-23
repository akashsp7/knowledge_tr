# Strict public read-only profile for Codex and local testing.

X_AUTH_MODE=bearer
X_BEARER_TOKEN=i8QEAAAAA0GTsAtzggCMS84rAhG36VMmETE0%3DJaPbWjKCN9rJS5nAkoElMkFt3PdEyCWn

# Force a narrow, explicit read-only tool surface.

X_READ_ONLY_ONLY=1
X_API_TOOL_ALLOWLIST=getPostsById,getPostsByIds,searchPostsRecent,searchPostsAll,getUsersById,getUsersByIds,getUsersByUsername,getUsersByUsernames,getUsersPosts,getUsersFollowers,getUsersFollowing,getListsById,getListsFollowers,getListsMembers,getListsPosts,getUsersOwnedLists,getUsersFollowedLists,getUsersListMemberships

# X API settings

X_API_BASE_URL=https://api.x.com
X_API_TIMEOUT=30
X_API_DEBUG=1

# MCP settings

MCP_TRANSPORT=stdio
MCP_HOST=127.0.0.1
MCP_PORT=8000
MCP_PATH=/mcp
