# Swarm MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that provides access to your [Foursquare Swarm](https://www.swarmapp.com/) check-in data. Use it with Claude Desktop, Claude Code, or any MCP-compatible client to analyze your check-in history.

## Features

| Tool | Description |
|------|-------------|
| `get_checkins` | Get paginated check-in history |
| `get_checkins_by_date_range` | Get check-ins within a specific date range |
| `get_recent_checkins` | Get check-ins from the past X days |
| `get_checkin_details` | Get details about a specific check-in |
| `get_all_checkins` | Retrieve your entire check-in history |
| `get_checkin_stats` | Get statistics (total count, date range, averages) |
| `search_checkins` | Search check-ins by venue name or category |

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
- "What's my most visited venue?"
- "Show me all my check-ins in New York"
- "What are my check-in stats?"
- "Search my check-ins for 'airport'"

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
