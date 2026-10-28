# swarm-mcp

An MCP server over a Foursquare Swarm check-in history, back to the first check-in.

## Status

Shipped and maintained. First released December 18, 2025, ten months before Foursquare shipped its own Swarm MCP in October 2026. Re-verified against the live Foursquare API on October 27, 2026: all 11 tools pass.

## License

MIT

## History

| Date | Release | What changed |
|------|---------|--------------|
| 2025-12-18 | [0.1.0](https://pypi.org/project/swarm-mcp/0.1.0/) | First public release on GitHub and PyPI: check-in history, date ranges, recent check-ins, stats, search |
| 2025-12-22 | [0.2.0](https://pypi.org/project/swarm-mcp/0.2.0/) | `_meta` transparency block and per-tool cost signals |
| 2025-12-22 | [0.3.0](https://pypi.org/project/swarm-mcp/0.3.0/) | Category, city, and date filters on search |
| 2025-12-22 | [0.4.0](https://pypi.org/project/swarm-mcp/0.4.0/) | `get_categories` with parent-category matching |
| 2025-12-23 | [0.5.0](https://pypi.org/project/swarm-mcp/0.5.0/) | Venue details, top venues, state and country filters |
| 2026-10-07 | [0.5.1](https://pypi.org/project/swarm-mcp/0.5.1/) | Pinned `mcp<2` after MCP Python SDK 2.0 removed the server API it used |
| 2026-10-27 | [0.6.0](https://pypi.org/project/swarm-mcp/0.6.0/) | Ported to MCP Python SDK 2.x; tool errors now set `isError` |

The [commit history](https://github.com/alexpriest/swarm-mcp/commits/main) and [PyPI release history](https://pypi.org/project/swarm-mcp/#history) are the public record.

## Features

| Tool | Description |
|------|-------------|
| `get_checkins` | Get paginated check-in history |
| `get_checkins_by_date_range` | Get check-ins within a specific date range |
| `get_recent_checkins` | Get check-ins from the past X days |
| `get_checkin_details` | Get details about a specific check-in |
| `get_venue_details` | Get venue info: ratings, hours, tips, photos |
| `get_all_checkins` | Retrieve your entire check-in history |
| `get_checkin_stats` | Get statistics (total count, date range, averages) |
| `get_categories` | List all unique categories in your history with counts |
| `get_top_venues` | Get your most visited venues with filters |
| `search_checkins` | Search with filters: query, category, city, state, country, date range |
| `get_server_info` | Get server metadata, data sources, and tool costs |

All responses include a `_meta` block with transparency info (completeness, API calls made, data scope) to help AI models make informed decisions about expensive operations.

### Tool Costs

| Tool | Cost | Notes |
|------|------|-------|
| `get_checkins` | Low | Single paginated request |
| `get_checkins_by_date_range` | Low | Single request with date filters |
| `get_recent_checkins` | Low | Single request with time filter |
| `get_checkin_details` | Low | Single check-in lookup |
| `get_venue_details` | Low | Single venue lookup |
| `get_checkin_stats` | Low | 2 API calls (newest + oldest) |
| `get_categories` | **High** | Scans history to discover unique categories |
| `get_top_venues` | **High** | Scans history to aggregate venue visits |
| `get_server_info` | None | Local introspection only |
| `get_all_checkins` | **High** | 1 API call per 250 check-ins |
| `search_checkins` | **High** | Client-side filtering; scans up to 5000 items |

### Example `_meta` Response

Every tool response includes metadata like this:

```json
{
  "_meta": {
    "is_complete": true,
    "returned_count": 50,
    "total_available": 1847,
    "limit_applied": 50,
    "api_calls_made": 1,
    "data_source": "foursquare_swarm_api",
    "data_scope": "authenticated_user_checkins"
  },
  "checkins": [...]
}
```

## Installation

### Using uvx (recommended)

```bash
uvx swarm-mcp
```

### Using pip

```bash
pip install swarm-mcp
```

## Setup

### 1. Get Your Foursquare Access Token

You'll need a Foursquare OAuth2 access token:

1. Go to [Foursquare Developer Apps](https://foursquare.com/developers/apps)
2. Create a new app (or use an existing one)
3. Note your **Client ID** and **Client Secret**
4. Generate an access token using the OAuth2 flow, or use the [API Explorer](https://docs.foursquare.com/developer/reference/v2-users-self) to get a token quickly

> **Security:** Treat `FOURSQUARE_TOKEN` like a password—don't commit it, paste it in issues, or share screenshots with it visible.

### 2. Configure Your MCP Client

#### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "swarm": {
      "command": "uvx",
      "args": ["swarm-mcp"],
      "env": {
        "FOURSQUARE_TOKEN": "your-access-token-here"
      }
    }
  }
}
```

#### Claude Code

```bash
claude mcp add swarm uvx swarm-mcp -e FOURSQUARE_TOKEN=your-access-token-here
```

Or add manually to your config:

```json
{
  "Swarm": {
    "command": "uvx",
    "args": ["swarm-mcp"],
    "env": {
      "FOURSQUARE_TOKEN": "your-access-token-here"
    }
  }
}
```

## Usage Examples

Once configured, you can ask Claude things like:

- "Show me my recent Swarm check-ins"
- "How many times have I checked into coffee shops this year?"
- "What are my top 10 most visited coffee shops?"
- "Show me all my check-ins in California"
- "What are my check-in stats?"
- "Get details about [venue name]" (ratings, hours, tips)
- "What are the ratings for my favorite restaurants?"

## Example Output

```
📊 SWARM CHECK-IN STATS
=============================================
Total check-ins:      12,456
Years active:         10.2 years
Days active:          3,726
Avg check-ins/day:    3.34

📅 First check-in: March 15, 2014 at Coffee Shop (NYC)
📍 Most recent: Today at Office (San Francisco)
```

## Development

```bash
# Clone the repo
git clone https://github.com/alexpriest/swarm-mcp.git
cd swarm-mcp

# Install in development mode
pip install -e .

# Run the server
FOURSQUARE_TOKEN=your-token swarm-mcp
```

## API Reference

This server uses the [Foursquare API v2](https://docs.foursquare.com/developer/reference/checkins):
- [User Check-ins](https://docs.foursquare.com/developer/reference/v2-users-checkins)
- [Check-in Details](https://docs.foursquare.com/developer/reference/v2-checkins-details)

## License

MIT License - see [LICENSE](LICENSE) for details.
