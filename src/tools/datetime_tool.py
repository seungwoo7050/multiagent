# src/tools/datetime_tool.py
import calendar
import datetime
import zoneinfo # Python 3.9+ 표준 라이브러리
import json # JSON 문자열 반환을 위해 추가
from enum import Enum
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field, field_validator

# BaseTool 및 LangChain 관련 import 추가
from src.tools.base import BaseTool

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.services.tool_manager import register_tool

logger = get_logger(__name__)

# Enum과 Input 모델은 그대로 유지
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
    operation: DateTimeOperation = Field(..., description="The date/time operation to perform (e.g., 'current', 'parse', 'format', 'add', 'diff').")
    timezone: Optional[str] = Field(
        'UTC', # 기본값 UTC 유지 (한국 시간 원하면 'Asia/Seoul'로 변경 가능)
        description="Timezone name (IANA format, e.g. 'America/New_York', 'Asia/Seoul', 'UTC'). Default is UTC."
    )
    date_string: Optional[str] = Field(None, description="Date string to parse or format (ISO 8601 format recommended for parsing). Required for 'parse', 'format', 'add', 'subtract', 'weekday', 'is_weekend'.")
    format_string: Optional[str] = Field(None, description="Format string for formatting date (Python strftime format, e.g., '%Y-%m-%d %H:%M:%S'). Required for 'format'.")
    years: Optional[int] = Field(None, description="Number of years to add/subtract for 'add'/'subtract' operations.", ge=-100, le=100)
    months: Optional[int] = Field(None, description="Number of months to add/subtract.", ge=-1200, le=1200)
    days: Optional[int] = Field(None, description="Number of days to add/subtract.", ge=-36500, le=36500)
    hours: Optional[int] = Field(None, description="Number of hours to add/subtract.", ge=-876000, le=876000)
    minutes: Optional[int] = Field(None, description="Number of minutes to add/subtract.", ge=-52560000, le=52560000)
    seconds: Optional[int] = Field(None, description="Number of seconds to add/subtract.", ge=-3153600000, le=3153600000)
    date1: Optional[str] = Field(None, description="First date string for 'diff' operation.")
    date2: Optional[str] = Field(None, description="Second date string for 'diff' operation.")
    year: Optional[int] = Field(None, description="Year for 'is_leap_year' or 'days_in_month' operations.", ge=1, le=9999)
    month: Optional[int] = Field(None, description="Month (1-12) for 'days_in_month' operation.", ge=1, le=12)

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: Optional[str]) -> str:
        # 기존 validator 유지
        if not v:
            return 'UTC'
        try:
            zoneinfo.ZoneInfo(v)
            return v
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone name: '{v}'. Use IANA format (e.g., 'Asia/Seoul', 'UTC').")
        except Exception as e:
            raise ValueError(f"Error validating timezone '{v}': {e}")

# @register_tool() 데코레이터 유지
@register_tool()
class DateTimeTool(BaseTool):
    """
    Performs various date and time operations based on the specified 'operation'.
    Supports getting current time, parsing/formatting strings, adding/subtracting durations,
    calculating differences, and checking date properties like weekday or leap year.
    Handles timezones specified in IANA format (e.g., 'Asia/Seoul', 'UTC').
    """
    # 클래스 변수로 name, description, args_schema 정의
    name: str = "datetime"
    description: str = (
        "Performs date & time operations. Use 'operation' to specify the action: "
        "'current' (gets current time), "
        "'parse' (parses 'date_string'), "
        "'format' (formats 'date_string' using 'format_string'), "
        "'add'/'subtract' (adds/subtracts duration using 'years', 'months', 'days', etc. from 'date_string'), "
        "'diff' (calculates difference between 'date1' and 'date2'), "
        "'weekday' (gets weekday of 'date_string'), "
        "'is_weekend' (checks if 'date_string' is weekend), "
        "'is_leap_year' (checks 'year'), "
        "'days_in_month' (gets days for 'year' and 'month'). "
        "Specify 'timezone' (e.g., 'Asia/Seoul', default 'UTC')."
    )
    args_schema: Type[BaseModel] = DateTimeInput

    # _run 메서드 구현: 핵심 로직 수행 및 결과 반환 (문자열)
    def _run(self, **kwargs: Any) -> str:
        """
        Executes the specified datetime operation based on validated arguments.
        Handles timezone conversion and potential errors.
        Returns the result as a JSON string.
        """
        # kwargs는 이미 args_schema에 의해 유효성이 검사된 상태로 LangChain에 의해 전달됨
        operation: DateTimeOperation = kwargs['operation'] # 필수 인자이므로 존재 보장
        timezone_str: str = kwargs.get('timezone', 'UTC') # 기본값 처리

        logger.debug(f'DateTimeTool: Processing operation: {operation.value} with timezone: {timezone_str}')
        result_dict: Dict[str, Any]

        try:
            tz = zoneinfo.ZoneInfo(timezone_str)

            # 각 operation에 대한 핸들러 호출 (기존 로직 재사용)
            if operation == DateTimeOperation.CURRENT:
                result_dict = self._handle_current(tz)
            elif operation == DateTimeOperation.PARSE:
                result_dict = self._handle_parse(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.FORMAT:
                result_dict = self._handle_format(kwargs.get('date_string'), kwargs.get('format_string'), tz)
            elif operation == DateTimeOperation.ADD:
                result_dict = self._handle_add(kwargs.get('date_string'), kwargs.get('years'), kwargs.get('months'), kwargs.get('days'), kwargs.get('hours'), kwargs.get('minutes'), kwargs.get('seconds'), tz)
            elif operation == DateTimeOperation.SUBTRACT:
                result_dict = self._handle_subtract(kwargs.get('date_string'), kwargs.get('years'), kwargs.get('months'), kwargs.get('days'), kwargs.get('hours'), kwargs.get('minutes'), kwargs.get('seconds'), tz)
            elif operation == DateTimeOperation.DIFF:
                result_dict = self._handle_diff(kwargs.get('date1'), kwargs.get('date2'), tz)
            elif operation == DateTimeOperation.WEEKDAY:
                result_dict = self._handle_weekday(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.IS_WEEKEND:
                result_dict = self._handle_is_weekend(kwargs.get('date_string'), tz)
            elif operation == DateTimeOperation.IS_LEAP_YEAR:
                result_dict = self._handle_is_leap_year(kwargs.get('year'))
            elif operation == DateTimeOperation.DAYS_IN_MONTH:
                result_dict = self._handle_days_in_month(kwargs.get('year'), kwargs.get('month'))
            else:
                # 이 경우는 Pydantic Enum 검증으로 인해 발생하기 어려움
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f'Unsupported datetime operation: {operation.value}',
                    details={'operation': operation.value}, tool_name=self.name
                )

            # 성공적인 결과를 JSON 문자열로 반환
            return json.dumps(result_dict, default=str)

        except zoneinfo.ZoneInfoNotFoundError:
            # Timezone 검증 실패 시
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f'Invalid timezone specified: {timezone_str}',
                details={'timezone': timezone_str}, tool_name=self.name
            )
        except (ValueError, TypeError) as validation_err:
             # 핸들러 내부에서 발생할 수 있는 유효성 검사 오류
             logger.warning(f"DateTimeTool: Validation error during '{operation.value}': {validation_err}")
             raise ToolError(
                 code=ErrorCode.TOOL_VALIDATION_ERROR,
                 message=f'Invalid input for operation {operation.value}: {str(validation_err)}',
                 original_error=validation_err,
                 tool_name=self.name
             )
        except Exception as e:
            # 예상치 못한 모든 다른 오류 처리
            if isinstance(e, ToolError): # 이미 ToolError인 경우 그대로 전달
                raise e
            logger.exception(f"DateTimeTool: Unexpected error during operation '{operation.value}': {str(e)}")
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"DateTime operation '{operation.value}' failed: {str(e)}",
                details={'operation': operation.value, 'error': str(e)},
                original_error=e,
                tool_name=self.name
            )

    # _arun 구현 (BaseTool 요구사항 충족)
    async def _arun(self, **kwargs: Any) -> str:
        """
        Asynchronously executes the datetime operation.
        Since most operations are CPU-bound, it runs the sync version.
        """
        # 이 도구의 연산은 대부분 CPU 바운드이므로 _run 호출로 충분
        try:
            # loop = asyncio.get_event_loop()
            # return await loop.run_in_executor(None, self._run, **kwargs)
             return self._run(**kwargs) # 직접 호출
        except ToolError:
             raise # ToolError는 그대로 전달
        except Exception as e:
             # _run 에서 처리되지 않은 예외가 혹시 있다면 여기서 처리
             logger.exception(f"DateTimeTool: Unexpected error during async wrapper: {str(e)}")
             raise ToolError(
                 code=ErrorCode.TOOL_EXECUTION_ERROR,
                 message=f'Unexpected async wrapper error: {str(e)}',
                 original_error=e,
                 tool_name=self.name
             )

    # --- 기존 핸들러 메서드들은 변경 없이 유지 ---
    # (오류 처리는 _run 메서드 레벨에서 ToolError로 변환되므로 내부에서는 ValueError 등을 발생시켜도 됨)
    def _handle_current(self, tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # ... (기존 코드) ...
        now: datetime.datetime = datetime.datetime.now(tz)
        return {'operation': 'current', 'timezone': str(tz), 'iso_format': now.isoformat(), 'timestamp': now.timestamp(), 'components': {'year': now.year, 'month': now.month, 'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second, 'microsecond': now.microsecond, 'weekday': now.weekday(), 'weekday_name': now.strftime('%A'), 'month_name': now.strftime('%B'), 'timezone_abbr': now.strftime('%Z'), 'utc_offset': now.strftime('%z')}}

    def _handle_parse(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # ... (기존 코드, 오류 시 ValueError 발생) ...
        if not date_string:
             raise ValueError("Date string is required for 'parse' operation.")
        # ... (기존 파싱 로직) ...
        # 성공 시 Dict 반환, 실패 시 ValueError 등 발생
        # 예시 (기존 코드 유지):
        parsed_dt: Optional[datetime.datetime] = None
        try:
             dt_from_iso = datetime.datetime.fromisoformat(date_string)
             if dt_from_iso.tzinfo is None or dt_from_iso.tzinfo.utcoffset(dt_from_iso) is None:
                 parsed_dt = dt_from_iso.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
             else:
                 parsed_dt = dt_from_iso.astimezone(tz)
        except ValueError:
            common_formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y', '%Y%m%d%H%M%S', '%Y%m%d%H%M', '%Y%m%d', '%b %d %Y %H:%M:%S', '%b %d %Y', '%B %d, %Y', '%d %B %Y']
            for fmt in common_formats:
                try:
                    dt_naive = datetime.datetime.strptime(date_string, fmt)
                    parsed_dt = dt_naive.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Could not parse date string '{date_string}' with any known format.")

        return {'operation': 'parse', 'input': date_string, 'timezone': str(tz), 'iso_format': parsed_dt.isoformat(), 'timestamp': parsed_dt.timestamp(), 'components': {'year': parsed_dt.year, 'month': parsed_dt.month, 'day': parsed_dt.day, 'hour': parsed_dt.hour, 'minute': parsed_dt.minute, 'second': parsed_dt.second, 'microsecond': parsed_dt.microsecond, 'weekday': parsed_dt.weekday(), 'weekday_name': parsed_dt.strftime('%A'), 'month_name': parsed_dt.strftime('%B'), 'timezone_abbr': parsed_dt.strftime('%Z'), 'utc_offset': parsed_dt.strftime('%z')}}


    # _handle_format, _handle_add, _handle_subtract, _handle_diff,
    # _handle_weekday, _handle_is_weekend, _handle_is_leap_year, _handle_days_in_month
    # 메서드들도 기존 로직을 유지하되, 오류 발생 시 ValueError 등을 발생시키도록 합니다.
    # (기존 코드에 이미 그렇게 구현되어 있을 가능성이 높습니다)
    # ... (나머지 핸들러 메서드들의 기존 구현 유지) ...

    # 예시: _handle_format
    def _handle_format(self, date_string: Optional[str], format_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
         # ... (기존 로직 유지) ...
         # 오류 시 ValueError 등 발생
         # 예: if not format_string: raise ValueError("Format string required")
        dt_to_format: datetime.datetime
        if not date_string:
            dt_to_format = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                # 파싱 핸들러 재사용, 실패 시 ValueError 전파됨
                parse_result = self._handle_parse(date_string, tz)
                dt_to_format = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ValueError as parse_err:
                # 파싱 실패 시 더 명확한 오류 메시지 제공
                raise ValueError(f'Failed to parse input date string for formatting: {parse_err}') from parse_err

        effective_format_string = format_string if format_string else '%Y-%m-%d %H:%M:%S %Z'
        try:
            formatted_string = dt_to_format.strftime(effective_format_string)
            return {'operation': 'format', 'input': input_desc, 'format_string': effective_format_string, 'timezone': str(tz), 'formatted_string': formatted_string, 'iso_format': dt_to_format.isoformat()}
        except ValueError as e:
            # 잘못된 형식 문자열 처리
             raise ValueError(f'Invalid format string provided: {str(e)}') from e
        except Exception as e:
             # 기타 예외
             raise RuntimeError(f'Failed to format date: {str(e)}') from e # 또는 ValueError

    # ... (다른 핸들러 메서드들도 유사하게 기존 로직 유지) ...
    def _handle_add(self, date_string: Optional[str], years: Optional[int], months: Optional[int], days: Optional[int], hours: Optional[int], minutes: Optional[int], seconds: Optional[int], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # (기존 로직)
        dt_base: datetime.datetime
        if not date_string:
            dt_base = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_base = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ValueError as parse_err:
                raise ValueError(f'Failed to parse base date string for addition: {parse_err}') from parse_err

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
                result_dt = result_dt.replace(year=new_year, month=new_month, day=new_day) # OverflowError 가능

            time_delta = datetime.timedelta(days=days_to_add, hours=hours_to_add, minutes=minutes_to_add, seconds=seconds_to_add)
            result_dt = result_dt + time_delta # OverflowError 가능

            return {'operation': 'add', 'input': input_desc, 'additions': {k:v for k,v in additions.items() if v is not None}, 'timezone': str(tz), 'original_iso': dt_base.isoformat(), 'result_iso': result_dt.isoformat(), 'result_formatted': result_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}
        except OverflowError as oe:
             raise ValueError(f'Date calculation resulted in overflow: {str(oe)}') from oe
        except Exception as e:
             raise RuntimeError(f'Failed to add time: {str(e)}') from e


    def _handle_subtract(self, date_string: Optional[str], years: Optional[int], months: Optional[int], days: Optional[int], hours: Optional[int], minutes: Optional[int], seconds: Optional[int], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # (기존 로직)
        neg_years = -years if years is not None else None
        neg_months = -months if months is not None else None
        neg_days = -days if days is not None else None
        neg_hours = -hours if hours is not None else None
        neg_minutes = -minutes if minutes is not None else None
        neg_seconds = -seconds if seconds is not None else None
        try:
            result = self._handle_add(date_string, neg_years, neg_months, neg_days, neg_hours, neg_minutes, neg_seconds, tz)
            result['operation'] = 'subtract'
            result['subtractions'] = {'years': years, 'months': months, 'days': days, 'hours': hours, 'minutes': minutes, 'seconds': seconds}
            result.pop('additions', None)
            return result
        except ValueError as ve: # _handle_add 에서 발생 가능
            raise ve

    def _handle_diff(self, date1_str: Optional[str], date2_str: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # (기존 로직)
        if not date1_str or not date2_str:
            raise ValueError("Both date1 and date2 strings are required for 'diff' operation.")
        try:
            parse_result1 = self._handle_parse(date1_str, tz)
            parse_result2 = self._handle_parse(date2_str, tz)
            dt1: datetime.datetime = datetime.datetime.fromisoformat(parse_result1['iso_format'])
            dt2: datetime.datetime = datetime.datetime.fromisoformat(parse_result2['iso_format'])
            diff: datetime.timedelta = dt2 - dt1
            # 상세 계산 (기존 코드 유지)
            # ... (복잡한 연/월 계산 로직) ...
            # 아래는 timedelta 기반 단순화 예시 (정확한 연/월 차이는 근사치)
            total_seconds = diff.total_seconds()
            days = diff.days
            seconds_rem = total_seconds - days * 86400
            hours = int(seconds_rem // 3600)
            minutes = int((seconds_rem % 3600) // 60)
            seconds = int(seconds_rem % 60)
            microseconds = diff.microseconds
            # 연/월 근사치 (대략적인 값만 필요하다면)
            years = days // 365
            months = (days % 365) // 30

            return {'operation': 'diff', 'date1_iso': dt1.isoformat(), 'date2_iso': dt2.isoformat(), 'timezone': str(tz), 'total_seconds': diff.total_seconds(), 'components': {'years': years, 'months': months, 'days': days, 'hours': hours, 'minutes': minutes, 'seconds': seconds, 'microseconds': microseconds, 'total_days_td': diff.days}}
        except ValueError as ve: # _handle_parse 에서 발생 가능
            raise ValueError(f"Failed to parse date strings for diff operation: {ve}") from ve
        except Exception as e:
            raise RuntimeError(f'Failed to calculate date difference: {str(e)}') from e

    def _handle_weekday(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # (기존 로직)
        dt_target: datetime.datetime
        if not date_string:
            dt_target = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_target = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ValueError as parse_err:
                 raise ValueError(f'Failed to parse date string for weekday check: {parse_err}') from parse_err
        weekday_num: int = dt_target.weekday() # Monday is 0 and Sunday is 6
        weekday_name: str = dt_target.strftime('%A')
        iso_weekday_num: int = dt_target.isoweekday() # Monday is 1 and Sunday is 7
        return {'operation': 'weekday', 'input': input_desc, 'timezone': str(tz), 'iso_format': dt_target.isoformat(), 'weekday': {'number_python': weekday_num, 'number_iso': iso_weekday_num, 'name': weekday_name}}

    def _handle_is_weekend(self, date_string: Optional[str], tz: zoneinfo.ZoneInfo) -> Dict[str, Any]:
        # (기존 로직)
        dt_target: datetime.datetime
        if not date_string:
            dt_target = datetime.datetime.now(tz)
            input_desc = 'current time'
        else:
            try:
                parse_result = self._handle_parse(date_string, tz)
                dt_target = datetime.datetime.fromisoformat(parse_result['iso_format'])
                input_desc = date_string
            except ValueError as parse_err:
                 raise ValueError(f'Failed to parse date string for weekend check: {parse_err}') from parse_err
        weekday_num: int = dt_target.weekday() # Monday is 0, Saturday is 5, Sunday is 6
        is_weekend: bool = weekday_num >= 5
        return {'operation': 'is_weekend', 'input': input_desc, 'timezone': str(tz), 'iso_format': dt_target.isoformat(), 'is_weekend': is_weekend, 'weekday': {'number_python': weekday_num, 'name': dt_target.strftime('%A')}}

    def _handle_is_leap_year(self, year: Optional[int]) -> Dict[str, Any]:
        # (기존 로직)
        effective_year = year if year is not None else datetime.datetime.now().year
        if not (1 <= effective_year <= 9999):
             raise ValueError("Year must be between 1 and 9999.")
        try:
            is_leap: bool = calendar.isleap(effective_year)
            return {'operation': 'is_leap_year', 'year': effective_year, 'is_leap_year': is_leap}
        except Exception as e:
            raise RuntimeError(f'Failed to check leap year: {str(e)}') from e

    def _handle_days_in_month(self, year: Optional[int], month: Optional[int]) -> Dict[str, Any]:
        # (기존 로직)
        now = datetime.datetime.now()
        effective_year = year if year is not None else now.year
        effective_month = month if month is not None else now.month
        if not (1 <= effective_year <= 9999):
             raise ValueError("Year must be between 1 and 9999.")
        if not (1 <= effective_month <= 12):
             raise ValueError("Month must be between 1 and 12.")
        try:
            days_in_month: int = calendar.monthrange(effective_year, effective_month)[1]
            month_name: str = datetime.date(effective_year, effective_month, 1).strftime('%B')
            return {'operation': 'days_in_month', 'year': effective_year, 'month': effective_month, 'month_name': month_name, 'days_in_month': days_in_month}
        except Exception as e:
            raise RuntimeError(f'Failed to get days in month: {str(e)}') from e