#!/usr/bin/env python3
"""
MCP Server for Foursquare Swarm check-in data.

Requires FOURSQUARE_TOKEN environment variable with OAuth2 access token.
Get your token from: https://foursquare.com/developers/apps
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Foursquare API configuration
API_BASE = "https://api.foursquare.com/v2"
API_VERSION = "20231010"  # Foursquare requires a version date

server = Server("swarm-mcp")


def get_token() -> str:
    """Get the Foursquare OAuth token from environment."""
    token = os.environ.get("FOURSQUARE_TOKEN")
    if not token:
        raise ValueError(
            "FOURSQUARE_TOKEN environment variable is required. "
            "Get your token from https://foursquare.com/developers/apps"
        )
    return token


async def make_request(endpoint: str, params: dict = None) -> dict:
    """Make an authenticated request to the Foursquare API."""
    token = get_token()

    url = f"{API_BASE}{endpoint}"
    request_params = {
        "oauth_token": token,
        "v": API_VERSION,
        **(params or {})
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=request_params, timeout=30.0)
        response.raise_for_status()
        return response.json()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_checkins",
            description="Get the authenticated user's check-in history. Returns check-ins with venue info, timestamps, and optional photos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of check-ins to return (max 250, default 50)",
                        "default": 50
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for pagination (default 0)",
                        "default": 0
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort order: 'newestfirst' or 'oldestfirst'",
                        "enum": ["newestfirst", "oldestfirst"],
                        "default": "newestfirst"
                    }
                }
            }
        ),
        Tool(
            name="get_checkins_by_date_range",
            description="Get check-ins within a specific date range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in ISO format (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in ISO format (YYYY-MM-DD)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of check-ins to return (max 250)",
                        "default": 250
                    }
                },
                "required": ["start_date", "end_date"]
            }
        ),
        Tool(
            name="get_recent_checkins",
            description="Get check-ins from the past X days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default 7)",
                        "default": 7
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of check-ins to return",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="get_checkin_details",
            description="Get detailed information about a specific check-in by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "checkin_id": {
                        "type": "string",
                        "description": "The ID of the check-in to retrieve"
                    }
                },
                "required": ["checkin_id"]
            }
        ),
        Tool(
            name="get_all_checkins",
            description="Get ALL check-ins by paginating through the entire history. Use with caution - may take time for users with many check-ins.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_checkins": {
                        "type": "integer",
                        "description": "Maximum total check-ins to retrieve (default 1000, use -1 for unlimited)",
                        "default": 1000
                    }
                }
            }
        ),
        Tool(
            name="get_checkin_stats",
            description="Get statistics about your check-in history (total count, date range, etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="search_checkins",
            description="Search check-ins by venue name or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to match against venue names"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        )
    ]


def format_checkin(checkin: dict) -> dict:
    """Format a check-in for readable output."""
    venue = checkin.get("venue", {})
    location = venue.get("location", {})

    # Get category
    categories = venue.get("categories", [])
    category = categories[0].get("name") if categories else "Unknown"

    # Format timestamp
    created_at = checkin.get("createdAt", 0)
    dt = datetime.fromtimestamp(created_at)

    return {
        "id": checkin.get("id"),
        "created_at": dt.isoformat(),
        "venue": {
            "name": venue.get("name", "Unknown"),
            "category": category,
            "address": location.get("formattedAddress", []),
            "city": location.get("city"),
            "state": location.get("state"),
            "country": location.get("country"),
            "lat": location.get("lat"),
            "lng": location.get("lng")
        },
        "shout": checkin.get("shout"),  # User's comment
        "photos_count": checkin.get("photos", {}).get("count", 0),
        "likes_count": checkin.get("likes", {}).get("count", 0),
        "comments_count": checkin.get("comments", {}).get("count", 0)
    }


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    try:
        if name == "get_checkins":
            limit = min(arguments.get("limit", 50), 250)
            offset = arguments.get("offset", 0)
            sort = arguments.get("sort", "newestfirst")

            data = await make_request("/users/self/checkins", {
                "limit": limit,
                "offset": offset,
                "sort": sort
            })

            checkins = data.get("response", {}).get("checkins", {})
            total = checkins.get("count", 0)
            items = [format_checkin(c) for c in checkins.get("items", [])]

            result = {
                "total_checkins": total,
                "returned": len(items),
                "offset": offset,
                "checkins": items
            }

        elif name == "get_checkins_by_date_range":
            start_date = datetime.fromisoformat(arguments["start_date"])
            end_date = datetime.fromisoformat(arguments["end_date"])
            limit = min(arguments.get("limit", 250), 250)

            # Convert to timestamps
            after_timestamp = int(start_date.timestamp())
            before_timestamp = int((end_date + timedelta(days=1)).timestamp())  # Include end date

            data = await make_request("/users/self/checkins", {
                "limit": limit,
                "afterTimestamp": after_timestamp,
                "beforeTimestamp": before_timestamp,
                "sort": "newestfirst"
            })

            checkins = data.get("response", {}).get("checkins", {})
            items = [format_checkin(c) for c in checkins.get("items", [])]

            result = {
                "date_range": f"{arguments['start_date']} to {arguments['end_date']}",
                "count": len(items),
                "checkins": items
            }

        elif name == "get_recent_checkins":
            days = arguments.get("days", 7)
            limit = min(arguments.get("limit", 50), 250)

            after_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())

            data = await make_request("/users/self/checkins", {
                "limit": limit,
                "afterTimestamp": after_timestamp,
                "sort": "newestfirst"
            })

            checkins = data.get("response", {}).get("checkins", {})
            items = [format_checkin(c) for c in checkins.get("items", [])]

            result = {
                "period": f"Last {days} days",
                "count": len(items),
                "checkins": items
            }

        elif name == "get_checkin_details":
            checkin_id = arguments["checkin_id"]

            data = await make_request(f"/checkins/{checkin_id}")
            checkin = data.get("response", {}).get("checkin", {})

            result = format_checkin(checkin)

            # Add photos if available
            photos = checkin.get("photos", {}).get("items", [])
            if photos:
                result["photos"] = [
                    {
                        "url": f"{p.get('prefix')}original{p.get('suffix')}",
                        "width": p.get("width"),
                        "height": p.get("height")
                    }
                    for p in photos
                ]

        elif name == "get_all_checkins":
            max_checkins = arguments.get("max_checkins", 1000)
            all_checkins = []
            offset = 0
            batch_size = 250

            while True:
                data = await make_request("/users/self/checkins", {
                    "limit": batch_size,
                    "offset": offset,
                    "sort": "newestfirst"
                })

                checkins = data.get("response", {}).get("checkins", {})
                items = checkins.get("items", [])

                if not items:
                    break

                all_checkins.extend([format_checkin(c) for c in items])
                offset += len(items)

                if max_checkins > 0 and len(all_checkins) >= max_checkins:
                    all_checkins = all_checkins[:max_checkins]
                    break

                if len(items) < batch_size:
                    break

            result = {
                "total_retrieved": len(all_checkins),
                "checkins": all_checkins
            }

        elif name == "get_checkin_stats":
            # Get first page to get total count
            data = await make_request("/users/self/checkins", {
                "limit": 1,
                "sort": "newestfirst"
            })

            total = data.get("response", {}).get("checkins", {}).get("count", 0)
            newest = data.get("response", {}).get("checkins", {}).get("items", [])

            # Get oldest check-in
            oldest_data = await make_request("/users/self/checkins", {
                "limit": 1,
                "sort": "oldestfirst"
            })
            oldest = oldest_data.get("response", {}).get("checkins", {}).get("items", [])

            result = {
                "total_checkins": total,
                "newest_checkin": format_checkin(newest[0]) if newest else None,
                "oldest_checkin": format_checkin(oldest[0]) if oldest else None,
            }

            if newest and oldest:
                newest_dt = datetime.fromtimestamp(newest[0].get("createdAt", 0))
                oldest_dt = datetime.fromtimestamp(oldest[0].get("createdAt", 0))
                days_active = (newest_dt - oldest_dt).days
                result["days_active"] = days_active
                result["avg_checkins_per_day"] = round(total / max(days_active, 1), 2)

        elif name == "search_checkins":
            query = arguments["query"].lower()
            limit = arguments.get("limit", 50)

            # We need to fetch check-ins and filter locally
            # The API doesn't support venue name search on check-ins
            all_checkins = []
            offset = 0
            batch_size = 250

            while len(all_checkins) < limit:
                data = await make_request("/users/self/checkins", {
                    "limit": batch_size,
                    "offset": offset,
                    "sort": "newestfirst"
                })

                checkins = data.get("response", {}).get("checkins", {})
                items = checkins.get("items", [])

                if not items:
                    break

                for item in items:
                    venue = item.get("venue", {})
                    venue_name = venue.get("name", "").lower()
                    categories = venue.get("categories", [])
                    category_names = [c.get("name", "").lower() for c in categories]

                    if query in venue_name or any(query in cat for cat in category_names):
                        all_checkins.append(format_checkin(item))
                        if len(all_checkins) >= limit:
                            break

                offset += len(items)

                if len(items) < batch_size:
                    break

                # Safety limit to avoid infinite loops
                if offset > 5000:
                    break

            result = {
                "query": arguments["query"],
                "count": len(all_checkins),
                "checkins": all_checkins
            }

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except httpx.HTTPStatusError as e:
        error_msg = f"API error: {e.response.status_code}"
        try:
            error_detail = e.response.json()
            error_msg += f" - {error_detail}"
        except:
            pass
        return [TextContent(type="text", text=error_msg)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Run the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
