import os
import time
import datetime
import calendar
import zoneinfo
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Literal, Type
from pydantic import BaseModel, Field, field_validator
from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.tools.base import BaseTool
from src.tools.registry import register_tool
logger = get_logger(__name__)

class DateTimeOperation(str, Enum):
    CURRENT = 'current'
    PARSE = 'parse'
    FORMAT = 'format'
    ADD = 'add'
    SUBTRACT = 'subtract'
    DIFF = 'diff'
    WEEKDAY = 'weekday'
    IS_WEEKEND = 'is_weekend'
    IS_LEAP_YEAR = 'is_leap_year'
    DAYS_IN_MONTH = 'days_in_month'

class DateTimeInput(BaseModel):
    operation: DateTimeOperation = Field(..., description='The date/time operation to perform')
    timezone: Optional[str] = Field('UTC', description="Timezone name (IANA format, e.g. 'America/New_York', 'Europe/London')")
    date_string: Optional[str] = Field(None, description='Date string to parse or format (ISO 8601 format recommended for parsing)')
    format_string: Optional[str] = Field(None, description='Format string for formatting date (Python strftime format)')
    years: Optional[int] = Field(None, description='Number of years to add/subtract', ge=-100, le=100)
    months: Optional[int] = Field(None, description='Number of months to add/subtract', ge=-1200, le=1200)
    days: Optional[int] = Field(None, description='Number of days to add/subtract', ge=-36500, le=36500)
    hours: Optional[int] = Field(None, description='Number of hours to add/subtract', ge=-876000, le=876000)
    minutes: Optional[int] = Field(None, description='Number of minutes to add/subtract', ge=-52560000, le=52560000)
    seconds: Optional[int] = Field(None, description='Number of seconds to add/subtract', ge=-3153600000, le=3153600000)
    date1: Optional[str] = Field(None, description='First date for difference calculation')
    date2: Optional[str] = Field(None, description='Second date for difference calculation')
    year: Optional[int] = Field(None, description='Year for operations like is_leap_year or days_in_month', ge=1, le=9999)
    month: Optional[int] = Field(None, description='Month for operations like days_in_month', ge=1, le=12)

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> str:
        if not v:
            return 'UTC'
        try:
            zoneinfo.ZoneInfo(v)
            return v
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone name: '{v}'. Please use IANA format (e.g., 'Asia/Seoul', 'UTC').")
        except Exception as e:
            raise ValueError(f"Error validating timezone '{v}': {e}")

@register_tool()
class DateTimeTool(BaseTool):

    @property
    def name(self) -> str:
        return 'datetime'

    @property
    def description(self) -> str:
        return 'Performs date and time operations like getting current time, parsing dates, formatting, addition, subtraction, and difference calculation.'

    @property
    def args_schema(self) -> Type[BaseModel]:
        return DateTimeInput

    def _run(self, **kwargs: Any) -> Any:
        return self._process_operation(**kwargs)

    async def _arun(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def _process_operation(self, **kwargs: Any) -> Dict[str, Any]:
        operation: DateTimeOperation = kwargs['operation']
        timezone_str: str = kwargs.get('timezone', 'UTC')
        logger.debug(f'Processing datetime operation: {operation.value} with timezone: {timezone_str}')
        try:
            tz = zoneinfo.ZoneInfo(timezone_str)
            if operation == DateTimeOperation.CURRENT:
                return self._handle_current(tz)
            elif operation == DateTimeOperation.PARSE:
                return self._handle_parse(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.FORMAT:
                return self._handle_format(kwargs.get('date_string'), kwargs.get('format_string'), tz)
            elif operation == DateTimeOperation.ADD:
                return self._handle_add(kwargs.get('date_string'), kwargs.get('years'), kwargs.get('months'), kwargs.get('days'), kwargs.get('hours'), kwargs.get('minutes'), kwargs.get('seconds'), tz)
            elif operation == DateTimeOperation.SUBTRACT:
                return self._handle_subtract(kwargs.get('date_string'), kwargs.get('years'), kwargs.get('months'), kwargs.get('days'), kwargs.get('hours'), kwargs.get('minutes'), kwargs.get('seconds'), tz)
            elif operation == DateTimeOperation.DIFF:
                return self._handle_diff(kwargs.get('date1'), kwargs.get('date2'), tz)
            elif operation == DateTimeOperation.WEEKDAY:
                return self._handle_weekday(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.IS_WEEKEND:
                return self._handle_is_weekend(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.IS_LEAP_YEAR:
                return self._handle_is_leap_year(kwargs.get('year'))
            elif operation == DateTimeOperation.DAYS_IN_MONTH:
                return self._handle_days_in_month(kwargs.get('year'), kwargs.get('month'))
            else:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Unsupported datetime operation: {operation.value}', details={'operation': operation.value}, tool_name=self.name)
        except zoneinfo.ZoneInfoNotFoundError:
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Invalid timezone specified: {timezone_str}', details={'timezone': timezone_str}, tool_name=self.name)
        except Exception as e:
            if isinstance(e, ToolError):
                raise e
            logger.error(f"DateTime operation '{operation.value}' failed: {str(e)}", extra={'operation': operation.value, 'error': str(e)}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f"DateTime operation '{operation.value}' failed: {str(e)}", details={'operation': operation.value, 'error': str(e)}, original_error=e, tool_name=self.name)

    def _handle_current(self, tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        now: datetime.datetime = datetime.datetime.now(tz)
        return {'operation': 'current', 'timezone': str(tz), 'iso_format': now.isoformat(), 'timestamp': now.timestamp(), 'components': {'year': now.year, 'month': now.month, 'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second, 'microsecond': now.microsecond, 'weekday': now.weekday(), 'weekday_name': now.strftime('%A'), 'month_name': now.strftime('%B'), 'timezone_abbr': now.strftime('%Z'), 'utc_offset': now.strftime('%z')}}

    def _handle_parse(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        if not date_string:
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message="Date string is required for 'parse' operation.", details={'operation': 'parse'}, tool_name=self.name)
        logger.debug(f"Attempting to parse date string: '{date_string}' with target timezone: {str(tz)}")
        parsed_dt: Optional[datetime.datetime] = None
        try:
            try:
                dt_from_iso = datetime.datetime.fromisoformat(date_string)
                if dt_from_iso.tzinfo is None or dt_from_iso.tzinfo.utcoffset(dt_from_iso) is None:
                    parsed_dt = dt_from_iso.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
                    logger.debug(f"Parsed '{date_string}' as naive ISO, assumed UTC, converted to {str(tz)}.")
                else:
                    parsed_dt = dt_from_iso.astimezone(tz)
                    logger.debug(f"Parsed '{date_string}' as aware ISO, converted to {str(tz)}.")
            except ValueError:
                logger.debug(f"Parsing '{date_string}' with fromisoformat failed. Trying strptime formats.")
                pass
            if parsed_dt is None:
                common_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y', '%Y%m%d%H%M%S', '%Y%m%d%H%M', '%Y%m%d', '%b %d %Y %H:%M:%S', '%b %d %Y', '%B %d, %Y', '%d %B %Y']
                for fmt in common_formats:
                    try:
                        dt_naive = datetime.datetime.strptime(date_string, fmt)
                        parsed_dt = dt_naive.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
                        logger.debug(f"Parsed '{date_string}' using format '{fmt}', assumed UTC, converted to {str(tz)}.")
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Could not parse date string '{date_string}' with any known format.")
            return {'operation': 'parse', 'input': date_string, 'timezone': str(tz), 'iso_format': parsed_dt.isoformat(), 'timestamp': parsed_dt.timestamp(), 'components': {'year': parsed_dt.year, 'month': parsed_dt.month, 'day': parsed_dt.day, 'hour': parsed_dt.hour, 'minute': parsed_dt.minute, 'second': parsed_dt.second, 'microsecond': parsed_dt.microsecond, 'weekday': parsed_dt.weekday(), 'weekday_name': parsed_dt.strftime('%A'), 'month_name': parsed_dt.strftime('%B'), 'timezone_abbr': parsed_dt.strftime('%Z'), 'utc_offset': parsed_dt.strftime('%z')}}
        except Exception as e:
            if isinstance(e, ToolError):
                raise e
            logger.error(f"Failed to parse date string '{date_string}': {str(e)}", exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to parse date string: {str(e)}', details={'date_string': date_string}, original_error=e, tool_name=self.name)

    def _handle_format(self, date_string: Optional[str], format_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        dt_to_format: datetime.datetime
        if not date_string:
            dt_to_format = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_to_format = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ToolError as parse_err:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to parse input date string for formatting: {parse_err.message}', details={'date_string': date_string}, original_error=parse_err.original_error, tool_name=self.name)
        effective_format_string = format_string if format_string else '%Y-%m-%d %H:%M:%S %Z'
        logger.debug(f"Formatting datetime '{dt_to_format.isoformat()}' using format '{effective_format_string}' and timezone '{str(tz)}'")
        try:
            formatted_string = dt_to_format.strftime(effective_format_string)
            return {'operation': 'format', 'input': input_desc, 'format_string': effective_format_string, 'timezone': str(tz), 'formatted_string': formatted_string, 'iso_format': dt_to_format.isoformat()}
        except ValueError as e:
            logger.error(f"Invalid format string used: '{effective_format_string}'. Error: {e}")
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Invalid format string provided: {str(e)}', details={'format_string': effective_format_string}, original_error=e, tool_name=self.name)
        except Exception as e:
            logger.error(f'Failed to format datetime: {e}', exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to format date: {str(e)}', details={'input': input_desc, 'format_string': effective_format_string}, original_error=e, tool_name=self.name)

    def _handle_add(self, date_string: Optional[str], years: Optional[int], months: Optional[int], days: Optional[int], hours: Optional[int], minutes: Optional[int], seconds: Optional[int], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        dt_base: datetime.datetime
        if not date_string:
            dt_base = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_base = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ToolError as parse_err:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to parse base date string for addition: {parse_err.message}', details={'date_string': date_string}, original_error=parse_err.original_error, tool_name=self.name)
        logger.debug(f'Adding time to base datetime: {dt_base.isoformat()}')
        additions = {'years': years, 'months': months, 'days': days, 'hours': hours, 'minutes': minutes, 'seconds': seconds}
        years_to_add = years or 0
        months_to_add = months or 0
        days_to_add = days or 0
        hours_to_add = hours or 0
        minutes_to_add = minutes or 0
        seconds_to_add = seconds or 0
        try:
            result_dt = dt_base
            if years_to_add != 0 or months_to_add != 0:
                total_months_delta = years_to_add * 12 + months_to_add
                new_year = result_dt.year + (result_dt.month - 1 + total_months_delta) // 12
                new_month = (result_dt.month - 1 + total_months_delta) % 12 + 1
                last_day_of_new_month = calendar.monthrange(new_year, new_month)[1]
                new_day = min(result_dt.day, last_day_of_new_month)
                result_dt = result_dt.replace(year=new_year, month=new_month, day=new_day)
                logger.debug(f'After adding {years_to_add} years and {months_to_add} months: {result_dt.isoformat()}')
            time_delta = datetime.timedelta(days=days_to_add, hours=hours_to_add, minutes=minutes_to_add, seconds=seconds_to_add)
            result_dt = result_dt + time_delta
            logger.debug(f'After adding time delta ({time_delta}): {result_dt.isoformat()}')
            return {'operation': 'add', 'input': input_desc, 'additions': additions, 'timezone': str(tz), 'original_iso': dt_base.isoformat(), 'result_iso': result_dt.isoformat(), 'result_formatted': result_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}
        except OverflowError as oe:
            logger.error(f'Date calculation resulted in overflow: {oe}')
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Date calculation resulted in overflow: {str(oe)}', original_error=oe, tool_name=self.name)
        except Exception as e:
            logger.error(f'Failed to add time: {str(e)}', exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to add time: {str(e)}', details={'base_date': dt_base.isoformat(), 'additions': additions}, original_error=e, tool_name=self.name)

    def _handle_subtract(self, date_string: Optional[str], years: Optional[int], months: Optional[int], days: Optional[int], hours: Optional[int], minutes: Optional[int], seconds: Optional[int], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        logger.debug('Handling subtract operation by negating values and calling add handler.')
        neg_years = -years if years is not None else None
        neg_months = -months if months is not None else None
        neg_days = -days if days is not None else None
        neg_hours = -hours if hours is not None else None
        neg_minutes = -minutes if minutes is not None else None
        neg_seconds = -seconds if seconds is not None else None
        result = self._handle_add(date_string, neg_years, neg_months, neg_days, neg_hours, neg_minutes, neg_seconds, tz)
        result['operation'] = 'subtract'
        result['subtractions'] = {'years': years, 'months': months, 'days': days, 'hours': hours, 'minutes': minutes, 'seconds': seconds}
        result.pop('additions', None)
        return result

    def _handle_diff(self, date1_str: Optional[str], date2_str: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        if not date1_str or not date2_str:
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message="Both date1 and date2 strings are required for 'diff' operation.", details={'date1': date1_str, 'date2': date2_str}, tool_name=self.name)
        logger.debug(f"Calculating difference between '{date1_str}' and '{date2_str}' in timezone '{str(tz)}'")
        try:
            parse_result1 = self._handle_parse(date1_str, tz)
            parse_result2 = self._handle_parse(date2_str, tz)
            dt1: datetime.datetime = datetime.datetime.fromisoformat(parse_result1['iso_format'])
            dt2: datetime.datetime = datetime.datetime.fromisoformat(parse_result2['iso_format'])
            diff: datetime.timedelta = dt2 - dt1
            years_diff = dt2.year - dt1.year
            months_diff = dt2.month - dt1.month
            if dt2.day < dt1.day:
                months_diff -= 1
            elif dt2.day == dt1.day and dt2.time() < dt1.time():
                months_diff -= 1
            total_months = years_diff * 12 + months_diff
            years = total_months // 12
            months = total_months % 12
            temp_date = dt1
            try:
                temp_date = temp_date.replace(year=temp_date.year + years)
            except ValueError:
                if temp_date.month == 2 and temp_date.day == 29:
                    temp_date = temp_date.replace(year=temp_date.year + years, day=28)
                else:
                    raise
            final_month = temp_date.month + months
            final_year = temp_date.year + final_month // 12
            final_month = final_month % 12
            if final_month == 0:
                final_month = 12
                final_year -= 1
            last_day_of_month = calendar.monthrange(final_year, final_month)[1]
            final_day = min(temp_date.day, last_day_of_month)
            try:
                temp_date = temp_date.replace(year=final_year, month=final_month, day=final_day)
            except ValueError as replace_err:
                logger.warning(f'Date replacement error during diff calculation: {replace_err}')
                years, months = (0, 0)
            remaining_timedelta: datetime.timedelta = dt2 - temp_date
            remaining_days = remaining_timedelta.days
            remaining_seconds = remaining_timedelta.seconds
            remaining_microseconds = remaining_timedelta.microseconds
            hours = remaining_seconds // 3600
            minutes = remaining_seconds % 3600 // 60
            seconds = remaining_seconds % 60
            return {'operation': 'diff', 'date1_iso': dt1.isoformat(), 'date2_iso': dt2.isoformat(), 'timezone': str(tz), 'total_seconds': diff.total_seconds(), 'components': {'years': years, 'months': months, 'days': remaining_days, 'hours': hours, 'minutes': minutes, 'seconds': seconds, 'microseconds': remaining_microseconds, 'total_days_td': diff.days}}
        except Exception as e:
            if isinstance(e, ToolError):
                raise e
            logger.error(f"Failed to calculate date difference between '{date1_str}' and '{date2_str}': {str(e)}", exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to calculate date difference: {str(e)}', details={'date1': date1_str, 'date2': date2_str}, original_error=e, tool_name=self.name)

    def _handle_weekday(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        dt_target: datetime.datetime
        if not date_string:
            dt_target = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_target = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ToolError as parse_err:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to parse date string for weekday check: {parse_err.message}', details={'date_string': date_string}, original_error=parse_err.original_error, tool_name=self.name)
        weekday_num: int = dt_target.weekday()
        weekday_name: str = dt_target.strftime('%A')
        iso_weekday_num: int = dt_target.isoweekday()
        logger.debug(f'Weekday for {dt_target.isoformat()}: {weekday_name} ({weekday_num})')
        return {'operation': 'weekday', 'input': input_desc, 'timezone': str(tz), 'iso_format': dt_target.isoformat(), 'weekday': {'number_python': weekday_num, 'number_iso': iso_weekday_num, 'name': weekday_name}}

    def _handle_is_weekend(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        dt_target: datetime.datetime
        if not date_string:
            dt_target = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_target = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ToolError as parse_err:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to parse date string for weekend check: {parse_err.message}', details={'date_string': date_string}, original_error=parse_err.original_error, tool_name=self.name)
        weekday_num: int = dt_target.weekday()
        is_weekend: bool = weekday_num >= 5
        logger.debug(f'Is weekend check for {dt_target.isoformat()}: {is_weekend} (Weekday: {weekday_num})')
        return {'operation': 'is_weekend', 'input': input_desc, 'timezone': str(tz), 'iso_format': dt_target.isoformat(), 'is_weekend': is_weekend, 'weekday': {'number_python': weekday_num, 'name': dt_target.strftime('%A')}}

    def _handle_is_leap_year(self, year: Optional[int]) -> Dict[str, Any]:
        effective_year = year if year is not None else datetime.datetime.now().year
        logger.debug(f'Checking if year {effective_year} is a leap year.')
        try:
            is_leap: bool = calendar.isleap(effective_year)
            return {'operation': 'is_leap_year', 'year': effective_year, 'is_leap_year': is_leap}
        except Exception as e:
            logger.error(f'Failed to check leap year for {effective_year}: {str(e)}', exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to check leap year: {str(e)}', details={'year': effective_year}, original_error=e, tool_name=self.name)

    def _handle_days_in_month(self, year: Optional[int], month: Optional[int]) -> Dict[str, Any]:
        now = datetime.datetime.now()
        effective_year = year if year is not None else now.year
        effective_month = month if month is not None else now.month
        logger.debug(f'Getting number of days in month: {effective_year}-{effective_month:02d}')
        try:
            days_in_month: int = calendar.monthrange(effective_year, effective_month)[1]
            month_name: str = datetime.date(effective_year, effective_month, 1).strftime('%B')
            return {'operation': 'days_in_month', 'year': effective_year, 'month': effective_month, 'month_name': month_name, 'days_in_month': days_in_month}
        except ValueError as ve:
            logger.error(f'Invalid year/month combination: {effective_year}-{effective_month}. Error: {ve}')
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Invalid month value: {effective_month}', details={'year': effective_year, 'month': effective_month}, original_error=ve, tool_name=self.name)
        except Exception as e:
            logger.error(f'Failed to get days in month for {effective_year}-{effective_month}: {str(e)}', exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to get days in month: {str(e)}', details={'year': effective_year, 'month': effective_month}, original_error=e, tool_name=self.name)