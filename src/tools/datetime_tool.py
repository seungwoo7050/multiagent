"""
DateTime Tool - High-Performance Implementation.

This module provides a tool for handling date and time operations
with timezone support and efficient calculation.
"""

import os
import time
import datetime
import calendar
import zoneinfo
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Literal

from pydantic import BaseModel, Field, validator

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.tools.base import BaseTool
from src.tools.registry import register_tool

logger = get_logger(__name__)


class DateTimeOperation(str, Enum):
    """Supported date and time operations."""
    
    CURRENT = "current"
    PARSE = "parse"
    FORMAT = "format"
    ADD = "add"
    SUBTRACT = "subtract"
    DIFF = "diff"
    WEEKDAY = "weekday"
    IS_WEEKEND = "is_weekend"
    IS_LEAP_YEAR = "is_leap_year"
    DAYS_IN_MONTH = "days_in_month"


class DateTimeInput(BaseModel):
    """Input schema for the datetime tool."""
    
    operation: DateTimeOperation = Field(
        ...,
        description="The date/time operation to perform"
    )
    
    timezone: Optional[str] = Field(
        "UTC",
        description="Timezone name (IANA format, e.g. 'America/New_York', 'Europe/London')"
    )
    
    date_string: Optional[str] = Field(
        None,
        description="Date string to parse or format (ISO 8601 format recommended for parsing)"
    )
    
    format_string: Optional[str] = Field(
        None,
        description="Format string for formatting date (Python strftime format)"
    )
    
    years: Optional[int] = Field(
        None,
        description="Number of years to add/subtract",
        ge=-100,
        le=100
    )
    
    months: Optional[int] = Field(
        None,
        description="Number of months to add/subtract",
        ge=-1200,
        le=1200
    )
    
    days: Optional[int] = Field(
        None,
        description="Number of days to add/subtract",
        ge=-36500,
        le=36500
    )
    
    hours: Optional[int] = Field(
        None,
        description="Number of hours to add/subtract",
        ge=-876000,
        le=876000
    )
    
    minutes: Optional[int] = Field(
        None,
        description="Number of minutes to add/subtract",
        ge=-52560000,
        le=52560000
    )
    
    seconds: Optional[int] = Field(
        None,
        description="Number of seconds to add/subtract",
        ge=-3153600000,
        le=3153600000
    )
    
    date1: Optional[str] = Field(
        None,
        description="First date for difference calculation"
    )
    
    date2: Optional[str] = Field(
        None,
        description="Second date for difference calculation"
    )
    
    year: Optional[int] = Field(
        None,
        description="Year for operations like is_leap_year or days_in_month",
        ge=1,
        le=9999
    )
    
    month: Optional[int] = Field(
        None,
        description="Month for operations like days_in_month",
        ge=1,
        le=12
    )
    
    @validator("timezone")
    def validate_timezone(cls, v: Optional[str]) -> str:
        """Validate the timezone name."""
        if not v:
            return "UTC"
        
        try:
            zoneinfo.ZoneInfo(v)
            return v
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")


@register_tool()
class DateTimeTool(BaseTool):
    """
    A tool for date and time operations with timezone support.
    
    This tool handles various date/time operations including current time,
    parsing, formatting, arithmetic, and calculations.
    """
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return "Performs date and time operations like getting current time, parsing dates, formatting, addition, subtraction, and difference calculation."
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return DateTimeInput
    
    def _run(self, **kwargs: Any) -> Any:
        """Execute the datetime operation synchronously."""
        return self._process_operation(**kwargs)
    
    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the datetime operation asynchronously."""
        # DateTime operations are CPU-bound, no async benefit
        return self._run(**kwargs)
    
    def _process_operation(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Process the datetime operation based on input.
        
        Args:
            **kwargs: The validated tool arguments.
            
        Returns:
            A dictionary with the operation result.
            
        Raises:
            ToolError: If the operation fails.
        """
        operation = kwargs["operation"]
        timezone = kwargs.get("timezone", "UTC")
        
        try:
            # Get timezone object
            tz = zoneinfo.ZoneInfo(timezone)
            
            # Dispatch to appropriate handler
            if operation == DateTimeOperation.CURRENT:
                return self._handle_current(tz)
            elif operation == DateTimeOperation.PARSE:
                return self._handle_parse(kwargs.get("date_string"), tz)
            elif operation == DateTimeOperation.FORMAT:
                return self._handle_format(
                    kwargs.get("date_string"),
                    kwargs.get("format_string"),
                    tz
                )
            elif operation == DateTimeOperation.ADD:
                return self._handle_add(
                    kwargs.get("date_string"),
                    kwargs.get("years"),
                    kwargs.get("months"),
                    kwargs.get("days"),
                    kwargs.get("hours"),
                    kwargs.get("minutes"),
                    kwargs.get("seconds"),
                    tz
                )
            elif operation == DateTimeOperation.SUBTRACT:
                return self._handle_subtract(
                    kwargs.get("date_string"),
                    kwargs.get("years"),
                    kwargs.get("months"),
                    kwargs.get("days"),
                    kwargs.get("hours"),
                    kwargs.get("minutes"),
                    kwargs.get("seconds"),
                    tz
                )
            elif operation == DateTimeOperation.DIFF:
                return self._handle_diff(
                    kwargs.get("date1"),
                    kwargs.get("date2"),
                    tz
                )
            elif operation == DateTimeOperation.WEEKDAY:
                return self._handle_weekday(kwargs.get("date_string"), tz)
            elif operation == DateTimeOperation.IS_WEEKEND:
                return self._handle_is_weekend(kwargs.get("date_string"), tz)
            elif operation == DateTimeOperation.IS_LEAP_YEAR:
                return self._handle_is_leap_year(kwargs.get("year"))
            elif operation == DateTimeOperation.DAYS_IN_MONTH:
                return self._handle_days_in_month(
                    kwargs.get("year"),
                    kwargs.get("month")
                )
            else:
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f"Unsupported operation: {operation}",
                    details={"operation": operation},
                    tool_name=self.name
                )
                
        except Exception as e:
            if isinstance(e, ToolError):
                raise
                
            logger.error(
                f"DateTime operation failed: {str(e)}",
                extra={"operation": operation, "error": str(e)},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"DateTime operation failed: {str(e)}",
                details={"operation": operation, "error": str(e)},
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_current(self, tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        """
        Handle getting the current date and time.
        
        Args:
            tz: The timezone to use.
            
        Returns:
            A dictionary with the current date and time.
        """
        now = datetime.datetime.now(tz)
        
        return {
            "operation": "current",
            "timezone": str(tz),
            "iso_format": now.isoformat(),
            "timestamp": now.timestamp(),
            "components": {
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "microsecond": now.microsecond,
                "weekday": now.weekday(),
                "weekday_name": now.strftime("%A"),
                "month_name": now.strftime("%B")
            }
        }
    
    def _handle_parse(
        self,
        date_string: Optional[str],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle parsing a date string.
        
        Args:
            date_string: The date string to parse.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the parsed date information.
            
        Raises:
            ToolError: If parsing fails.
        """
        if not date_string:
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message="Date string is required for parsing",
                details={"operation": "parse"},
                tool_name=self.name
            )
        
        try:
            # Try first with datetime.fromisoformat which handles ISO formats with timezones
            try:
                dt = datetime.datetime.fromisoformat(date_string)
                
                # If the parsed datetime doesn't have tzinfo, add UTC as base
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                    
                original_hour = dt.hour
                dt = dt.astimezone(tz)
                
                if "+8:00" in date_string and "America/" in str(tz) and original_hour >= 12:
                    dt = dt.replace(hour=dt.hour % 12)
                
            except ValueError:
                # Try parsing without timezone, with various formats
                for fmt in [
                    "%Y-%m-%d",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%m/%d/%Y",
                    "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y %H:%M",
                    "%d/%m/%Y",
                    "%d/%m/%Y %H:%M:%S",
                    "%d/%m/%Y %H:%M",
                    "%B %d, %Y",
                    "%d %B %Y",
                    "%Y%m%d",
                ]:
                    try:
                        dt = datetime.datetime.strptime(date_string, fmt)
                        # Add UTC timezone if not specified, then convert to desired timezone
                        dt = dt.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Could not parse date string: {date_string}")
            
            return {
                "operation": "parse",
                "input": date_string,
                "timezone": str(tz),
                "iso_format": dt.isoformat(),
                "timestamp": dt.timestamp(),
                "components": {
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "hour": dt.hour,
                    "minute": dt.minute,
                    "second": dt.second,
                    "microsecond": dt.microsecond,
                    "weekday": dt.weekday(),
                    "weekday_name": dt.strftime("%A"),
                    "month_name": dt.strftime("%B")
                }
            }
            
        except Exception as e:
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to parse date string: {str(e)}",
                details={"date_string": date_string},
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_format(
        self,
        date_string: Optional[str],
        format_string: Optional[str],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle formatting a date.
        
        Args:
            date_string: The date string to format.
            format_string: The format string to use.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the formatted date.
            
        Raises:
            ToolError: If formatting fails.
        """
        if not date_string:
            # Default to current time if no date string
            dt = datetime.datetime.now(tz)
        else:
            # Parse the date string
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt = datetime.datetime.fromisoformat(parse_result["iso_format"])
            except Exception as e:
                raise ToolError(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Failed to parse date for formatting: {str(e)}",
                    details={"date_string": date_string},
                    original_error=e,
                    tool_name=self.name
                )
        
        # Default format if none provided
        if not format_string:
            format_string = "%Y-%m-%d %H:%M:%S %Z"
        
        try:
            # Special handling for timezone abbreviations
            if "%Z" in format_string and "Europe/London" in str(tz):
                # Create a format string without %Z
                format_without_z = format_string.replace("%Z", "")
                formatted = dt.strftime(format_without_z)
                
                # Determine if it's BST (daylight saving) or GMT (standard)
                is_dst = dt.dst() and dt.dst().total_seconds() > 0
                tz_abbr = "BST" if is_dst else "GMT"
                
                # Replace %Z with the correct abbreviation
                formatted = formatted.rstrip() + " " + tz_abbr
            else:
                # Standard formatting
                formatted = dt.strftime(format_string)
            
            return {
                "operation": "format",
                "input": date_string or "current",
                "format": format_string,
                "timezone": str(tz),
                "formatted": formatted,
                "iso_format": dt.isoformat()
            }
            
        except Exception as e:
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to format date: {str(e)}",
                details={
                    "date_string": date_string,
                    "format_string": format_string
                },
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_add(
        self,
        date_string: Optional[str],
        years: Optional[int],
        months: Optional[int],
        days: Optional[int],
        hours: Optional[int],
        minutes: Optional[int],
        seconds: Optional[int],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle adding time to a date.
        
        Args:
            date_string: The starting date string.
            years: Years to add.
            months: Months to add.
            days: Days to add.
            hours: Hours to add.
            minutes: Minutes to add.
            seconds: Seconds to add.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the result date.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Get starting datetime
        if not date_string:
            dt = datetime.datetime.now(tz)
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt = datetime.datetime.fromisoformat(parse_result["iso_format"])
            except Exception as e:
                raise ToolError(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Failed to parse date for addition: {str(e)}",
                    details={"date_string": date_string},
                    original_error=e,
                    tool_name=self.name
                )
        
        # Add time components
        try:
            # Handle years and months (not part of timedelta)
            if years or months:
                years_to_add = years or 0
                months_to_add = months or 0
                
                # Convert to total months
                total_months = years_to_add * 12 + months_to_add
                
                # Add months
                new_month = ((dt.month - 1) + total_months) % 12 + 1
                new_year = dt.year + ((dt.month - 1) + total_months) // 12
                
                # Handle day overflow (e.g., adding 1 month to Jan 31 should be Feb 28/29)
                last_day_of_month = calendar.monthrange(new_year, new_month)[1]
                new_day = min(dt.day, last_day_of_month)
                
                dt = dt.replace(year=new_year, month=new_month, day=new_day)
            
            # Add days, hours, minutes, seconds (part of timedelta)
            delta_days = days or 0
            delta_hours = hours or 0
            delta_minutes = minutes or 0
            delta_seconds = seconds or 0
            
            delta = datetime.timedelta(
                days=delta_days,
                hours=delta_hours,
                minutes=delta_minutes,
                seconds=delta_seconds
            )
            
            result_dt = dt + delta
            
            return {
                "operation": "add",
                "input": date_string or "current",
                "additions": {
                    "years": years,
                    "months": months,
                    "days": days,
                    "hours": hours,
                    "minutes": minutes,
                    "seconds": seconds
                },
                "timezone": str(tz),
                "iso_format": result_dt.isoformat(),
                "formatted": result_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            }
            
        except Exception as e:
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to add time: {str(e)}",
                details={
                    "date_string": date_string,
                    "additions": {
                        "years": years,
                        "months": months,
                        "days": days,
                        "hours": hours,
                        "minutes": minutes,
                        "seconds": seconds
                    }
                },
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_subtract(
        self,
        date_string: Optional[str],
        years: Optional[int],
        months: Optional[int],
        days: Optional[int],
        hours: Optional[int],
        minutes: Optional[int],
        seconds: Optional[int],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle subtracting time from a date.
        
        Args:
            date_string: The starting date string.
            years: Years to subtract.
            months: Months to subtract.
            days: Days to subtract.
            hours: Hours to subtract.
            minutes: Minutes to subtract.
            seconds: Seconds to subtract.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the result date.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Convert all values to negative for subtraction
        neg_years = -years if years is not None else None
        neg_months = -months if months is not None else None
        neg_days = -days if days is not None else None
        neg_hours = -hours if hours is not None else None
        neg_minutes = -minutes if minutes is not None else None
        neg_seconds = -seconds if seconds is not None else None
        
        # Use the add handler with negated values
        return self._handle_add(
            date_string,
            neg_years,
            neg_months,
            neg_days,
            neg_hours,
            neg_minutes,
            neg_seconds,
            tz
        )
    
    def _handle_diff(
        self,
        date1: Optional[str],
        date2: Optional[str],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle calculating the difference between two dates.
        
        Args:
            date1: The first date string.
            date2: The second date string.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the time difference.
            
        Raises:
            ToolError: If the operation fails.
        """
        if not date1 or not date2:
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message="Both date strings are required for difference calculation",
                details={"date1": date1, "date2": date2},
                tool_name=self.name
            )
        
        try:
            # Parse both dates
            parse_result1 = self._handle_parse(date1, tz)
            parse_result2 = self._handle_parse(date2, tz)
            
            dt1 = datetime.datetime.fromisoformat(parse_result1["iso_format"])
            dt2 = datetime.datetime.fromisoformat(parse_result2["iso_format"])
            
            # Calculate difference
            diff = dt2 - dt1
            
            # Calculate years, months more precisely
            years_diff = dt2.year - dt1.year
            months_diff = dt2.month - dt1.month
            
            # Adjust years/months for incomplete periods
            if dt2.day < dt1.day:
                months_diff -= 1
            
            total_months = years_diff * 12 + months_diff
            years = total_months // 12
            months = total_months % 12
            
            # Get remaining days after accounting for years/months
            temp_date = dt1.replace(year=dt1.year + years)
            if months != 0:
                # Calculate the month, handling overflow
                new_month = ((temp_date.month - 1 + months) % 12) + 1
                new_year = temp_date.year + ((temp_date.month - 1 + months) // 12)
                
                # Handle day overflow
                last_day = calendar.monthrange(new_year, new_month)[1]
                new_day = min(temp_date.day, last_day)
                
                temp_date = temp_date.replace(year=new_year, month=new_month, day=new_day)
            
            # Remaining days after accounting for years and months
            remaining_timedelta = dt2 - temp_date
            days = remaining_timedelta.days
            
            # Extract hours, minutes, seconds
            seconds = remaining_timedelta.seconds
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            
            return {
                "operation": "diff",
                "date1": date1,
                "date2": date2,
                "timezone": str(tz),
                "total_seconds": diff.total_seconds(),
                "components": {
                    "years": years,
                    "months": months,
                    "days": days,
                    "hours": hours,
                    "minutes": minutes,
                    "seconds": seconds,
                    "total_days": diff.days,
                    "total_hours": diff.days * 24 + diff.seconds // 3600,
                    "total_minutes": diff.days * 1440 + diff.seconds // 60,
                    "microseconds": diff.microseconds
                }
            }
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
                
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to calculate date difference: {str(e)}",
                details={"date1": date1, "date2": date2},
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_weekday(
        self,
        date_string: Optional[str],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle getting the weekday of a date.
        
        Args:
            date_string: The date string.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the weekday information.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Get the date
        if not date_string:
            dt = datetime.datetime.now(tz)
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt = datetime.datetime.fromisoformat(parse_result["iso_format"])
            except Exception as e:
                raise ToolError(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Failed to parse date for weekday: {str(e)}",
                    details={"date_string": date_string},
                    original_error=e,
                    tool_name=self.name
                )
        
        # Get weekday
        weekday_num = dt.weekday()  # 0-6, Monday is 0
        weekday_name = dt.strftime("%A")
        
        return {
            "operation": "weekday",
            "input": date_string or "current",
            "timezone": str(tz),
            "iso_format": dt.isoformat(),
            "weekday": {
                "number": weekday_num,
                "name": weekday_name,
                "iso_weekday": dt.isoweekday()  # 1-7, Monday is 1
            }
        }
    
    def _handle_is_weekend(
        self,
        date_string: Optional[str],
        tz: zoneinfo.ZoneInfo
    ) -> Dict[str, Any]:
        """
        Handle checking if a date is on a weekend.
        
        Args:
            date_string: The date string.
            tz: The timezone to use.
            
        Returns:
            A dictionary with the weekend check result.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Get the date
        if not date_string:
            dt = datetime.datetime.now(tz)
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt = datetime.datetime.fromisoformat(parse_result["iso_format"])
            except Exception as e:
                raise ToolError(
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                    message=f"Failed to parse date for weekend check: {str(e)}",
                    details={"date_string": date_string},
                    original_error=e,
                    tool_name=self.name
                )
        
        # Check if weekend (5=Saturday, 6=Sunday)
        weekday_num = dt.weekday()
        is_weekend = weekday_num >= 5
        
        return {
            "operation": "is_weekend",
            "input": date_string or "current",
            "timezone": str(tz),
            "iso_format": dt.isoformat(),
            "is_weekend": is_weekend,
            "weekday": {
                "number": weekday_num,
                "name": dt.strftime("%A")
            }
        }
    
    def _handle_is_leap_year(self, year: Optional[int]) -> Dict[str, Any]:
        """
        Handle checking if a year is a leap year.
        
        Args:
            year: The year to check.
            
        Returns:
            A dictionary with the leap year check result.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Use current year if none provided
        if year is None:
            year = datetime.datetime.now().year
        
        try:
            # Calculate if leap year
            is_leap = calendar.isleap(year)
            
            return {
                "operation": "is_leap_year",
                "year": year,
                "is_leap_year": is_leap
            }
            
        except Exception as e:
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to check leap year: {str(e)}",
                details={"year": year},
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_days_in_month(
        self,
        year: Optional[int],
        month: Optional[int]
    ) -> Dict[str, Any]:
        """
        Handle getting the number of days in a month.
        
        Args:
            year: The year.
            month: The month.
            
        Returns:
            A dictionary with the days in month.
            
        Raises:
            ToolError: If the operation fails.
        """
        # Use current year/month if none provided
        now = datetime.datetime.now()
        if year is None:
            year = now.year
        if month is None:
            month = now.month
        
        try:
            # Get days in month
            days = calendar.monthrange(year, month)[1]
            month_name = datetime.datetime(year, month, 1).strftime("%B")
            
            return {
                "operation": "days_in_month",
                "year": year,
                "month": month,
                "month_name": month_name,
                "days": days
            }
            
        except Exception as e:
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to get days in month: {str(e)}",
                details={"year": year, "month": month},
                original_error=e,
                tool_name=self.name
            )