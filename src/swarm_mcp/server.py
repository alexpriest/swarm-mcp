#!/usr/bin/env python3
"""
MCP Server for Foursquare Swarm check-in data.

Requires FOURSQUARE_TOKEN environment variable with OAuth2 access token.
Get your token from: https://foursquare.com/developers/apps
"""

import os
import asyncio
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


def build_meta(
    *,
    is_complete: bool,
    returned_count: int,
    total_available: int | None = None,
    limit_applied: int | None = None,
    truncated_reason: str | None = None,
    api_calls_made: int = 1,
    items_scanned: int | None = None,
) -> dict:
    """Build a standardized _meta object for response transparency."""
    meta = {
        "is_complete": is_complete,
        "returned_count": returned_count,
        "total_available": total_available,
        "api_calls_made": api_calls_made,
        "data_source": "foursquare_swarm_api",
        "data_scope": "authenticated_user_checkins",
    }
    if limit_applied is not None:
        meta["limit_applied"] = limit_applied
    if truncated_reason:
        meta["truncated_reason"] = truncated_reason
    if items_scanned is not None:
        meta["items_scanned"] = items_scanned
    return meta


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
            description="Get ALL check-ins by paginating through the entire history. EXPENSIVE: Makes 1 API call per 250 check-ins (e.g., 5000 check-ins = 20 API calls). Prefer get_checkins with manual pagination for incremental access.",
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
            description="Search check-ins with flexible filters. EXPENSIVE: Requires client-side filtering. Use filters to narrow results. For comprehensive searches, increase max_scan.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to match against venue names or categories (case-insensitive substring). Optional if using other filters."
                    },
                    "category": {
                        "type": "string",
                        "description": "Exact category match (e.g., 'Coffee Shop', 'Airport', 'Bar'). Case-insensitive."
                    },
                    "city": {
                        "type": "string",
                        "description": "Filter by city name (case-insensitive substring match)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in ISO format (YYYY-MM-DD). Only return check-ins on or after this date."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in ISO format (YYYY-MM-DD). Only return check-ins on or before this date."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 50)",
                        "default": 50
                    },
                    "max_scan": {
                        "type": "integer",
                        "description": "Maximum items to scan (default 5000). Increase for comprehensive searches, use -1 for unlimited.",
                        "default": 5000
                    }
                }
            }
        ),
        Tool(
            name="get_server_info",
            description="Get information about this MCP server: data sources, privacy scope, available tools, and their costs. Cheap introspection call with no external API requests.",
            inputSchema={
                "type": "object",
                "properties": {}
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

            remaining = total - offset - len(items)
            result = {
                "_meta": build_meta(
                    is_complete=(remaining <= 0),
                    returned_count=len(items),
                    total_available=total,
                    limit_applied=limit,
                    truncated_reason="limit_reached" if remaining > 0 else None,
                ),
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

            # If we got exactly `limit` items, there may be more
            possibly_truncated = len(items) >= limit
            result = {
                "_meta": build_meta(
                    is_complete=not possibly_truncated,
                    returned_count=len(items),
                    total_available=None,  # Unknown for date-filtered queries
                    limit_applied=limit,
                    truncated_reason="limit_reached" if possibly_truncated else None,
                ),
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

            possibly_truncated = len(items) >= limit
            result = {
                "_meta": build_meta(
                    is_complete=not possibly_truncated,
                    returned_count=len(items),
                    total_available=None,
                    limit_applied=limit,
                    truncated_reason="limit_reached" if possibly_truncated else None,
                ),
                "period": f"Last {days} days",
                "count": len(items),
                "checkins": items
            }

        elif name == "get_checkin_details":
            checkin_id = arguments["checkin_id"]

            data = await make_request(f"/checkins/{checkin_id}")
            checkin = data.get("response", {}).get("checkin", {})

            formatted = format_checkin(checkin)

            # Add photos if available
            photos = checkin.get("photos", {}).get("items", [])
            if photos:
                formatted["photos"] = [
                    {
                        "url": f"{p.get('prefix')}original{p.get('suffix')}",
                        "width": p.get("width"),
                        "height": p.get("height")
                    }
                    for p in photos
                ]

            result = {
                "_meta": build_meta(
                    is_complete=True,
                    returned_count=1,
                ),
                "checkin": formatted
            }

        elif name == "get_all_checkins":
            max_checkins = arguments.get("max_checkins", 1000)
            all_checkins = []
            offset = 0
            batch_size = 250
            api_calls = 0
            total_available = None
            hit_max_limit = False

            while True:
                data = await make_request("/users/self/checkins", {
                    "limit": batch_size,
                    "offset": offset,
                    "sort": "newestfirst"
                })
                api_calls += 1

                checkins = data.get("response", {}).get("checkins", {})
                if total_available is None:
                    total_available = checkins.get("count", 0)
                items = checkins.get("items", [])

                if not items:
                    break

                all_checkins.extend([format_checkin(c) for c in items])
                offset += len(items)

                if max_checkins > 0 and len(all_checkins) >= max_checkins:
                    all_checkins = all_checkins[:max_checkins]
                    hit_max_limit = True
                    break

                if len(items) < batch_size:
                    break

            is_complete = not hit_max_limit and len(all_checkins) >= (total_available or 0)
            result = {
                "_meta": build_meta(
                    is_complete=is_complete,
                    returned_count=len(all_checkins),
                    total_available=total_available,
                    limit_applied=max_checkins if max_checkins > 0 else None,
                    truncated_reason="max_checkins_reached" if hit_max_limit else None,
                    api_calls_made=api_calls,
                ),
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

            stats = {
                "total_checkins": total,
                "newest_checkin": format_checkin(newest[0]) if newest else None,
                "oldest_checkin": format_checkin(oldest[0]) if oldest else None,
            }

            if newest and oldest:
                newest_dt = datetime.fromtimestamp(newest[0].get("createdAt", 0))
                oldest_dt = datetime.fromtimestamp(oldest[0].get("createdAt", 0))
                days_active = (newest_dt - oldest_dt).days
                stats["days_active"] = days_active
                stats["avg_checkins_per_day"] = round(total / max(days_active, 1), 2)

            result = {
                "_meta": build_meta(
                    is_complete=True,
                    returned_count=1,
                    total_available=total,
                    api_calls_made=2,
                ),
                **stats
            }

        elif name == "search_checkins":
            # Extract filter parameters
            query = arguments.get("query", "").lower() if arguments.get("query") else None
            category_filter = arguments.get("category", "").lower() if arguments.get("category") else None
            city_filter = arguments.get("city", "").lower() if arguments.get("city") else None
            start_date_str = arguments.get("start_date")
            end_date_str = arguments.get("end_date")
            limit = arguments.get("limit", 50)
            max_scan = arguments.get("max_scan", 5000)

            # Validate that at least one filter is provided
            if not any([query, category_filter, city_filter, start_date_str, end_date_str]):
                result = {"error": "At least one filter (query, category, city, start_date, end_date) is required"}
            else:
                # Parse dates if provided
                after_timestamp = None
                before_timestamp = None
                if start_date_str:
                    start_date = datetime.fromisoformat(start_date_str)
                    after_timestamp = int(start_date.timestamp())
                if end_date_str:
                    end_date = datetime.fromisoformat(end_date_str)
                    before_timestamp = int((end_date + timedelta(days=1)).timestamp())

                # We need to fetch check-ins and filter locally
                matching_checkins = []
                offset = 0
                batch_size = 250
                api_calls = 0
                items_scanned = 0
                total_available = None
                hit_scan_limit = False
                exhausted_all = False

                while len(matching_checkins) < limit:
                    request_params = {
                        "limit": batch_size,
                        "offset": offset,
                        "sort": "newestfirst"
                    }
                    # Use API-level date filtering when possible (more efficient)
                    if after_timestamp:
                        request_params["afterTimestamp"] = after_timestamp
                    if before_timestamp:
                        request_params["beforeTimestamp"] = before_timestamp

                    data = await make_request("/users/self/checkins", request_params)
                    api_calls += 1

                    checkins = data.get("response", {}).get("checkins", {})
                    if total_available is None:
                        total_available = checkins.get("count", 0)
                    items = checkins.get("items", [])

                    if not items:
                        exhausted_all = True
                        break

                    for item in items:
                        items_scanned += 1
                        venue = item.get("venue", {})
                        venue_name = venue.get("name", "").lower()
                        location = venue.get("location", {})
                        venue_city = location.get("city", "").lower() if location.get("city") else ""
                        categories = venue.get("categories", [])
                        category_names = [c.get("name", "").lower() for c in categories]

                        # Apply filters (all specified filters must match)
                        matches = True

                        # Text query: substring match on venue name or category
                        if query and not (query in venue_name or any(query in cat for cat in category_names)):
                            matches = False

                        # Category filter: exact match on any category
                        if category_filter and not any(category_filter == cat for cat in category_names):
                            matches = False

                        # City filter: substring match
                        if city_filter and city_filter not in venue_city:
                            matches = False

                        if matches:
                            matching_checkins.append(format_checkin(item))
                            if len(matching_checkins) >= limit:
                                break

                    offset += len(items)

                    if len(items) < batch_size:
                        exhausted_all = True
                        break

                    # Safety limit to avoid runaway searches (unless unlimited)
                    if max_scan > 0 and offset >= max_scan:
                        hit_scan_limit = True
                        break

                # Determine completeness
                is_complete = exhausted_all or len(matching_checkins) >= limit
                truncated_reason = None
                if hit_scan_limit:
                    pct = round(items_scanned / total_available * 100) if total_available else 0
                    truncated_reason = f"scan_limit_reached ({items_scanned:,} of {total_available:,} scanned, {pct}%)"
                elif len(matching_checkins) >= limit and not exhausted_all:
                    truncated_reason = "result_limit_reached"

                # Build filters summary for response
                filters_applied = {}
                if query:
                    filters_applied["query"] = arguments.get("query")
                if category_filter:
                    filters_applied["category"] = arguments.get("category")
                if city_filter:
                    filters_applied["city"] = arguments.get("city")
                if start_date_str:
                    filters_applied["start_date"] = start_date_str
                if end_date_str:
                    filters_applied["end_date"] = end_date_str

                result = {
                    "_meta": build_meta(
                        is_complete=is_complete and not hit_scan_limit,
                        returned_count=len(matching_checkins),
                        total_available=total_available,
                        limit_applied=limit,
                        truncated_reason=truncated_reason,
                        api_calls_made=api_calls,
                        items_scanned=items_scanned,
                    ),
                    "filters": filters_applied,
                    "count": len(matching_checkins),
                    "checkins": matching_checkins
                }

        elif name == "get_server_info":
            result = {
                "_meta": build_meta(
                    is_complete=True,
                    returned_count=1,
                    api_calls_made=0,
                ),
                "server": {
                    "name": "swarm-mcp",
                    "version": "0.1.0",
                    "description": "MCP server for Foursquare Swarm check-in data",
                },
                "data_source": {
                    "api": "Foursquare API v2",
                    "base_url": "https://api.foursquare.com/v2",
                    "authentication": "OAuth2 token (user-provided)",
                },
                "privacy": {
                    "data_scope": "Authenticated user's own check-ins only",
                    "data_flow": "Direct API calls to Foursquare; no data stored or cached by this server",
                    "third_party_access": "None; data flows directly between Foursquare and the MCP client",
                },
                "tools": {
                    "get_checkins": {"cost": "low", "api_calls": 1, "notes": "Single paginated request"},
                    "get_checkins_by_date_range": {"cost": "low", "api_calls": 1, "notes": "Single request with date filters"},
                    "get_recent_checkins": {"cost": "low", "api_calls": 1, "notes": "Single request with time filter"},
                    "get_checkin_details": {"cost": "low", "api_calls": 1, "notes": "Single check-in lookup"},
                    "get_all_checkins": {"cost": "high", "api_calls": "1 per 250 check-ins", "notes": "Paginated fetch of entire history"},
                    "get_checkin_stats": {"cost": "low", "api_calls": 2, "notes": "Fetches newest and oldest check-ins"},
                    "search_checkins": {"cost": "high", "api_calls": "1 per 250 scanned", "notes": "Supports query, category, city, date range filters. Use max_scan=-1 for comprehensive searches."},
                    "get_server_info": {"cost": "none", "api_calls": 0, "notes": "Local introspection only"},
                },
            }

        else:
            result = {"error": f"Unknown tool: {name}"}

        import json
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
