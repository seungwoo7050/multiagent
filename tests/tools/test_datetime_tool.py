"""Unit tests for the datetime tool."""

import pytest
import datetime
import zoneinfo
from pydantic import ValidationError

from src.config.errors import ToolError
from src.tools.datetime_tool import DateTimeTool, DateTimeInput, DateTimeOperation


@pytest.fixture
def datetime_tool():
    """Create a datetime tool for tests."""
    return DateTimeTool()


def test_datetime_initialization(datetime_tool):
    """Test datetime tool initialization."""
    assert datetime_tool.name == "datetime"
    assert "date and time" in datetime_tool.description.lower()
    assert datetime_tool.args_schema == DateTimeInput


def test_datetime_input_validation():
    """Test datetime input validation."""
    # Valid inputs for different operations
    valid_current = DateTimeInput(operation=DateTimeOperation.CURRENT)
    assert valid_current.operation == DateTimeOperation.CURRENT
    assert valid_current.timezone == "UTC"
    
    valid_parse = DateTimeInput(
        operation=DateTimeOperation.PARSE,
        date_string="2023-01-01"
    )
    assert valid_parse.operation == DateTimeOperation.PARSE
    assert valid_parse.date_string == "2023-01-01"
    
    # Invalid timezone
    with pytest.raises(ValidationError):
        DateTimeInput(operation=DateTimeOperation.CURRENT, timezone="Invalid/Zone")
    
    # Out of range values
    with pytest.raises(ValidationError):
        DateTimeInput(
            operation=DateTimeOperation.ADD,
            date_string="2023-01-01",
            years=101  # Max is 100
        )
    
    with pytest.raises(ValidationError):
        DateTimeInput(
            operation=DateTimeOperation.SUBTRACT,
            date_string="2023-01-01",
            days=-36501  # Min is -36500
        )


def test_current_operation(datetime_tool):
    """Test the current time operation."""
    result = datetime_tool.run(operation=DateTimeOperation.CURRENT)
    
    # Verify structure
    assert result["operation"] == "current"
    assert "timezone" in result
    assert "iso_format" in result
    assert "timestamp" in result
    assert "components" in result
    
    # Verify components
    components = result["components"]
    assert "year" in components
    assert "month" in components
    assert "day" in components
    assert "hour" in components
    assert "minute" in components
    assert "second" in components
    assert "weekday" in components
    assert "weekday_name" in components
    assert "month_name" in components
    
    # Value should be recent
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    result_time = result["timestamp"]
    assert abs(now - result_time) < 5  # Within 5 seconds


def test_parse_operation(datetime_tool):
    """Test the parse operation."""
    # ISO format
    result = datetime_tool.run(
        operation=DateTimeOperation.PARSE,
        date_string="2023-01-15T12:30:45Z"
    )
    
    assert result["operation"] == "parse"
    assert result["input"] == "2023-01-15T12:30:45Z"
    assert result["components"]["year"] == 2023
    assert result["components"]["month"] == 1
    assert result["components"]["day"] == 15
    assert result["components"]["hour"] == 12
    assert result["components"]["minute"] == 30
    assert result["components"]["second"] == 45
    
    # Common formats
    formats = [
        "2023-01-15",
        "01/15/2023",
        "15/01/2023",
        "January 15, 2023",
        "15 January 2023",
        "20230115"
    ]
    
    for date_str in formats:
        result = datetime_tool.run(
            operation=DateTimeOperation.PARSE,
            date_string=date_str
        )
        assert result["components"]["year"] == 2023
        assert result["components"]["month"] == 1
        assert result["components"]["day"] == 15
    
    # With timezone
    result = datetime_tool.run(
        operation=DateTimeOperation.PARSE,
        date_string="2023-01-15T12:30:45+08:00",
        timezone="America/New_York"
    )
    
    # Should convert to America/New_York timezone
    assert "America/New_York" in result["timezone"]
    # 12:30:45+08:00 is 04:30:45 UTC, which is earlier in NY
    assert result["components"]["hour"] < 12


def test_parse_operation_errors(datetime_tool):
    """Test parse operation with errors."""
    # Missing date_string
    with pytest.raises(ToolError) as excinfo:
        datetime_tool.run(operation=DateTimeOperation.PARSE)
    assert "required for parsing" in str(excinfo.value)
    
    # Invalid date string
    with pytest.raises(ToolError) as excinfo:
        datetime_tool.run(
            operation=DateTimeOperation.PARSE,
            date_string="not-a-date"
        )
    assert "Failed to parse date" in str(excinfo.value)


def test_format_operation(datetime_tool):
    """Test the format operation."""
    # With default format
    result = datetime_tool.run(
        operation=DateTimeOperation.FORMAT,
        date_string="2023-01-15"
    )
    
    assert result["operation"] == "format"
    assert result["input"] == "2023-01-15"
    assert "formatted" in result
    assert "iso_format" in result
    
    # With custom format
    result = datetime_tool.run(
        operation=DateTimeOperation.FORMAT,
        date_string="2023-01-15",
        format_string="%Y/%m/%d"
    )
    
    assert result["formatted"] == "2023/01/15"
    
    # With timezone
    result = datetime_tool.run(
        operation=DateTimeOperation.FORMAT,
        date_string="2023-01-15",
        timezone="Europe/London",
        format_string="%Y-%m-%d %H:%M:%S %Z"
    )
    
    assert "Europe/London" in result["timezone"]
    assert "GMT" in result["formatted"] or "BST" in result["formatted"]
    
    # Current time if no date provided
    result = datetime_tool.run(
        operation=DateTimeOperation.FORMAT,
        format_string="%Y"
    )
    
    current_year = datetime.datetime.now().year
    assert result["formatted"] == str(current_year)


def test_add_operation(datetime_tool):
    """Test the add operation."""
    # Add days
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-15",
        days=10
    )
    
    assert result["operation"] == "add"
    assert result["input"] == "2023-01-15"
    assert "2023-01-25" in result["iso_format"]
    
    # Add months
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-15",
        months=3
    )
    
    assert "2023-04-15" in result["iso_format"]
    
    # Add years
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-15",
        years=2
    )
    
    assert "2025-01-15" in result["iso_format"]
    
    # Add time components
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-15T12:00:00",
        hours=3,
        minutes=30,
        seconds=45
    )
    
    assert "2023-01-15T15:30:45" in result["iso_format"]
    
    # Add everything
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-15T12:00:00",
        years=1,
        months=2,
        days=3,
        hours=4,
        minutes=5,
        seconds=6
    )
    
    assert "2024-03-18T16:05:06" in result["iso_format"]
    
    # Handle month overflow (e.g., Jan 31 + 1 month)
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-31",
        months=1
    )
    
    assert "2023-02-28" in result["iso_format"]  # Feb 2023 has 28 days
    
    # Handle leap years
    result = datetime_tool.run(
        operation=DateTimeOperation.ADD,
        date_string="2023-01-31",
        years=1,
        months=1
    )
    
    assert "2024-02-29" in result["iso_format"]  # Feb 2024 is leap year


def test_subtract_operation(datetime_tool):
    """Test the subtract operation."""
    # Subtract days
    result = datetime_tool.run(
        operation=DateTimeOperation.SUBTRACT,
        date_string="2023-01-15",
        days=5
    )
    
    assert result["operation"] == "add"  # Internally uses add with negative values
    assert result["input"] == "2023-01-15"
    assert "2023-01-10" in result["iso_format"]
    
    # Subtract months
    result = datetime_tool.run(
        operation=DateTimeOperation.SUBTRACT,
        date_string="2023-03-15",
        months=2
    )
    
    assert "2023-01-15" in result["iso_format"]
    
    # Subtract years
    result = datetime_tool.run(
        operation=DateTimeOperation.SUBTRACT,
        date_string="2023-01-15",
        years=3
    )
    
    assert "2020-01-15" in result["iso_format"]
    
    # Subtract time components
    result = datetime_tool.run(
        operation=DateTimeOperation.SUBTRACT,
        date_string="2023-01-15T12:00:00",
        hours=6,
        minutes=30,
        seconds=15
    )
    
    assert "2023-01-15T05:29:45" in result["iso_format"]


def test_diff_operation(datetime_tool):
    """Test the diff operation."""
    # Basic difference
    result = datetime_tool.run(
        operation=DateTimeOperation.DIFF,
        date1="2023-01-01",
        date2="2023-01-10"
    )
    
    assert result["operation"] == "diff"
    assert result["date1"] == "2023-01-01"
    assert result["date2"] == "2023-01-10"
    assert result["components"]["days"] == 9
    
    # Years, months, days
    result = datetime_tool.run(
        operation=DateTimeOperation.DIFF,
        date1="2020-01-15",
        date2="2023-05-20"
    )
    
    assert result["components"]["years"] == 3
    assert result["components"]["months"] == 4
    assert result["components"]["days"] > 0  # Exact day depends on implementation
    
    # Time components
    result = datetime_tool.run(
        operation=DateTimeOperation.DIFF,
        date1="2023-01-01T12:00:00",
        date2="2023-01-01T15:30:45"
    )
    
    assert result["components"]["hours"] == 3
    assert result["components"]["minutes"] == 30
    assert result["components"]["seconds"] == 45
    
    # Negative difference (date2 before date1)
    result = datetime_tool.run(
        operation=DateTimeOperation.DIFF,
        date1="2023-01-10",
        date2="2023-01-01"
    )
    
    # Total seconds should be negative
    assert result["total_seconds"] < 0
    
    # Error case - missing dates
    with pytest.raises(ToolError) as excinfo:
        datetime_tool.run(
            operation=DateTimeOperation.DIFF,
            date1="2023-01-01"
            # date2 missing
        )
    assert "Both date strings are required" in str(excinfo.value)


def test_weekday_operation(datetime_tool):
    """Test the weekday operation."""
    # Sunday
    result = datetime_tool.run(
        operation=DateTimeOperation.WEEKDAY,
        date_string="2023-01-01"  # This was a Sunday
    )
    
    assert result["operation"] == "weekday"
    assert result["input"] == "2023-01-01"
    assert result["weekday"]["number"] == 6  # 0-based, Sunday is 6
    assert result["weekday"]["name"] == "Sunday"
    
    # Monday
    result = datetime_tool.run(
        operation=DateTimeOperation.WEEKDAY,
        date_string="2023-01-02"  # This was a Monday
    )
    
    assert result["weekday"]["number"] == 0  # 0-based, Monday is 0
    assert result["weekday"]["name"] == "Monday"
    
    # ISO weekday (1-7, Monday is 1)
    assert result["weekday"]["iso_weekday"] == 1


def test_is_weekend_operation(datetime_tool):
    """Test the is_weekend operation."""
    # Weekend (Sunday)
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_WEEKEND,
        date_string="2023-01-01"  # This was a Sunday
    )
    
    assert result["operation"] == "is_weekend"
    assert result["input"] == "2023-01-01"
    assert result["is_weekend"] is True
    
    # Weekend (Saturday)
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_WEEKEND,
        date_string="2023-01-07"  # This was a Saturday
    )
    
    assert result["is_weekend"] is True
    
    # Weekday
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_WEEKEND,
        date_string="2023-01-02"  # This was a Monday
    )
    
    assert result["is_weekend"] is False


def test_is_leap_year_operation(datetime_tool):
    """Test the is_leap_year operation."""
    # Leap year
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_LEAP_YEAR,
        year=2024
    )
    
    assert result["operation"] == "is_leap_year"
    assert result["year"] == 2024
    assert result["is_leap_year"] is True
    
    # Non-leap year
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_LEAP_YEAR,
        year=2023
    )
    
    assert result["is_leap_year"] is False
    
    # Century year (not leap)
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_LEAP_YEAR,
        year=2100
    )
    
    assert result["is_leap_year"] is False
    
    # Century year divisible by 400 (leap)
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_LEAP_YEAR,
        year=2000
    )
    
    assert result["is_leap_year"] is True
    
    # Current year if not specified
    result = datetime_tool.run(
        operation=DateTimeOperation.IS_LEAP_YEAR
    )
    
    current_year = datetime.datetime.now().year
    assert result["year"] == current_year


def test_days_in_month_operation(datetime_tool):
    """Test the days_in_month operation."""
    # 31-day month
    result = datetime_tool.run(
        operation=DateTimeOperation.DAYS_IN_MONTH,
        year=2023,
        month=1  # January
    )
    
    assert result["operation"] == "days_in_month"
    assert result["year"] == 2023
    assert result["month"] == 1
    assert result["month_name"] == "January"
    assert result["days"] == 31
    
    # 30-day month
    result = datetime_tool.run(
        operation=DateTimeOperation.DAYS_IN_MONTH,
        year=2023,
        month=4  # April
    )
    
    assert result["days"] == 30
    
    # February in non-leap year
    result = datetime_tool.run(
        operation=DateTimeOperation.DAYS_IN_MONTH,
        year=2023,
        month=2  # February
    )
    
    assert result["days"] == 28
    
    # February in leap year
    result = datetime_tool.run(
        operation=DateTimeOperation.DAYS_IN_MONTH,
        year=2024,
        month=2  # February
    )
    
    assert result["days"] == 29
    
    # Current year/month if not specified
    result = datetime_tool.run(
        operation=DateTimeOperation.DAYS_IN_MONTH
    )
    
    now = datetime.datetime.now()
    assert result["year"] == now.year
    assert result["month"] == now.month


@pytest.mark.asyncio
async def test_datetime_async(datetime_tool):
    """Test the async execution path."""
    # DateTime doesn't have a true async implementation, but we test the interface
    result = await datetime_tool.arun(
        operation=DateTimeOperation.CURRENT
    )
    
    assert result["operation"] == "current"
    assert "iso_format" in result
    assert "timestamp" in result


def test_invalid_operation(datetime_tool):
    """Test handling of invalid operations."""
    # Should never happen with enum validation, but test the error path
    with pytest.raises(ToolError) as excinfo:
        # Bypassing the enum validation in the pydantic model
        datetime_tool._process_operation(operation="invalid_op")
    
    assert "Unsupported operation" in str(excinfo.value)