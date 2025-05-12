A3='Year must be between 1 and 9999.'
A2='number_python'
A1='additions'
A0='%Y-%m-%d %H:%M:%S %Z'
z='utc_offset'
y='timezone_abbr'
x='weekday_name'
w='microsecond'
v='second'
u='minute'
t='hour'
s='day'
r='timestamp'
q='format_string'
p='diff'
o='subtract'
n='add'
m='format'
l='parse'
k='current'
j='%B'
i='month_name'
h='components'
g='UTC'
f='days_in_month'
e='is_leap_year'
d='is_weekend'
c=int
a='current time'
Z='%A'
Y='month'
O='input'
V='seconds'
U='minutes'
T='hours'
S='days'
R='months'
Q='years'
N=RuntimeError
M='year'
K='weekday'
J=Exception
I='timezone'
G='iso_format'
F='operation'
C=str
B=ValueError
A=None
import calendar as b,datetime as D,zoneinfo as W,json
from enum import Enum
from typing import Any,Dict,Optional,Type
from pydantic import BaseModel as A4,Field as E,field_validator as A5
from src.tools.base import BaseTool as A6
from src.config.errors import ErrorCode as P,ToolError as L
from src.utils.logger import get_logger as A7
from src.services.tool_manager import register_tool as A8
X=A7(__name__)
class H(C,Enum):CURRENT=k;PARSE=l;FORMAT=m;ADD=n;SUBTRACT=o;DIFF=p;WEEKDAY=K;IS_WEEKEND=d;IS_LEAP_YEAR=e;DAYS_IN_MONTH=f
class A9(A4):
	operation=E(...,description="The date/time operation to perform (e.g., 'current', 'parse', 'format', 'add', 'diff').");timezone=E(g,description="Timezone name (IANA format, e.g. 'America/New_York', 'Asia/Seoul', 'UTC'). Default is UTC.");date_string=E(A,description="Date string to parse or format (ISO 8601 format recommended for parsing). Required for 'parse', 'format', 'add', 'subtract', 'weekday', 'is_weekend'.");format_string=E(A,description="Format string for formatting date (Python strftime format, e.g., '%Y-%m-%d %H:%M:%S'). Required for 'format'.");years=E(A,description="Number of years to add/subtract for 'add'/'subtract' operations.",ge=-100,le=100);months=E(A,description='Number of months to add/subtract.',ge=-1200,le=1200);days=E(A,description='Number of days to add/subtract.',ge=-36500,le=36500);hours=E(A,description='Number of hours to add/subtract.',ge=-876000,le=876000);minutes=E(A,description='Number of minutes to add/subtract.',ge=-52560000,le=52560000);seconds=E(A,description='Number of seconds to add/subtract.',ge=-3153600000,le=3153600000);date1=E(A,description="First date string for 'diff' operation.");date2=E(A,description="Second date string for 'diff' operation.");year=E(A,description="Year for 'is_leap_year' or 'days_in_month' operations.",ge=1,le=9999);month=E(A,description="Month (1-12) for 'days_in_month' operation.",ge=1,le=12)
	@A5(I)
	@classmethod
	def validate_timezone(cls,v):
		if not v:return g
		try:W.ZoneInfo(v);return v
		except W.ZoneInfoNotFoundError:raise B(f"Invalid timezone name: '{v}'. Use IANA format (e.g., 'Asia/Seoul', 'UTC').")
		except J as A:raise B(f"Error validating timezone '{v}': {A}")
@A8()
class AA(A6):
	name='datetime';description="Performs date & time operations. Use 'operation' to specify the action: 'current' (gets current time), 'parse' (parses 'date_string'), 'format' (formats 'date_string' using 'format_string'), 'add'/'subtract' (adds/subtracts duration using 'years', 'months', 'days', etc. from 'date_string'), 'diff' (calculates difference between 'date1' and 'date2'), 'weekday' (gets weekday of 'date_string'), 'is_weekend' (checks if 'date_string' is weekend), 'is_leap_year' (checks 'year'), 'days_in_month' (gets days for 'year' and 'month'). Specify 'timezone' (e.g., 'Asia/Seoul', default 'UTC').";args_schema=A9
	def _run(E,**A):
		O='date_string';D=A[F];Z=A.get(I,g);X.debug(f"DateTimeTool: Processing operation: {D.value} with timezone: {Z}");G:0
		try:
			K=W.ZoneInfo(Z)
			if D==H.CURRENT:G=E._handle_current(K)
			elif D==H.PARSE:G=E._handle_parse(A.get(O),K)
			elif D==H.FORMAT:G=E._handle_format(A.get(O),A.get(q),K)
			elif D==H.ADD:G=E._handle_add(A.get(O),A.get(Q),A.get(R),A.get(S),A.get(T),A.get(U),A.get(V),K)
			elif D==H.SUBTRACT:G=E._handle_subtract(A.get(O),A.get(Q),A.get(R),A.get(S),A.get(T),A.get(U),A.get(V),K)
			elif D==H.DIFF:G=E._handle_diff(A.get('date1'),A.get('date2'),K)
			elif D==H.WEEKDAY:G=E._handle_weekday(A.get(O),K)
			elif D==H.IS_WEEKEND:G=E._handle_is_weekend(A.get(O),K)
			elif D==H.IS_LEAP_YEAR:G=E._handle_is_leap_year(A.get(M))
			elif D==H.DAYS_IN_MONTH:G=E._handle_days_in_month(A.get(M),A.get(Y))
			else:raise L(code=P.TOOL_VALIDATION_ERROR,message=f"Unsupported datetime operation: {D.value}",details={F:D.value},tool_name=E.name)
			return json.dumps(G,default=C)
		except W.ZoneInfoNotFoundError:raise L(code=P.TOOL_VALIDATION_ERROR,message=f"Invalid timezone specified: {Z}",details={I:Z},tool_name=E.name)
		except(B,TypeError)as a:X.warning(f"DateTimeTool: Validation error during '{D.value}': {a}");raise L(code=P.TOOL_VALIDATION_ERROR,message=f"Invalid input for operation {D.value}: {C(a)}",original_error=a,tool_name=E.name)
		except J as N:
			if isinstance(N,L):raise N
			X.exception(f"DateTimeTool: Unexpected error during operation '{D.value}': {C(N)}");raise L(code=P.TOOL_EXECUTION_ERROR,message=f"DateTime operation '{D.value}' failed: {C(N)}",details={F:D.value,'error':C(N)},original_error=N,tool_name=E.name)
	async def _arun(B,**D):
		try:return B._run(**D)
		except L:raise
		except J as A:X.exception(f"DateTimeTool: Unexpected error during async wrapper: {C(A)}");raise L(code=P.TOOL_EXECUTION_ERROR,message=f"Unexpected async wrapper error: {C(A)}",original_error=A,tool_name=B.name)
	def _handle_current(B,tz):A=D.datetime.now(tz);return{F:k,I:C(tz),G:A.isoformat(),r:A.timestamp(),h:{M:A.year,Y:A.month,s:A.day,t:A.hour,u:A.minute,v:A.second,w:A.microsecond,K:A.weekday(),x:A.strftime(Z),i:A.strftime(j),y:A.strftime('%Z'),z:A.strftime('%z')}}
	def _handle_parse(Q,date_string,tz):
		H=date_string
		if not H:raise B("Date string is required for 'parse' operation.")
		E=A
		try:
			J=D.datetime.fromisoformat(H)
			if J.tzinfo is A or J.tzinfo.utcoffset(J)is A:E=J.replace(tzinfo=D.timezone.utc).astimezone(tz)
			else:E=J.astimezone(tz)
		except B:
			L=['%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M','%Y-%m-%d','%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M','%m/%d/%Y','%d/%m/%Y %H:%M:%S','%d/%m/%Y %H:%M','%d/%m/%Y','%Y%m%d%H%M%S','%Y%m%d%H%M','%Y%m%d','%b %d %Y %H:%M:%S','%b %d %Y','%B %d, %Y','%d %B %Y']
			for N in L:
				try:P=D.datetime.strptime(H,N);E=P.replace(tzinfo=D.timezone.utc).astimezone(tz);break
				except B:continue
			else:raise B(f"Could not parse date string '{H}' with any known format.")
		return{F:l,O:H,I:C(tz),G:E.isoformat(),r:E.timestamp(),h:{M:E.year,Y:E.month,s:E.day,t:E.hour,u:E.minute,v:E.second,w:E.microsecond,K:E.weekday(),x:E.strftime(Z),i:E.strftime(j),y:E.strftime('%Z'),z:E.strftime('%z')}}
	def _handle_format(Q,date_string,format_string,tz):
		K=format_string;H=date_string;E:0
		if not H:E=D.datetime.now(tz);L=a
		else:
			try:R=Q._handle_parse(H,tz);E=D.datetime.fromisoformat(R[G]);L=H
			except B as M:raise B(f"Failed to parse input date string for formatting: {M}")from M
		P=K if K else A0
		try:S=E.strftime(P);return{F:m,O:L,q:P,I:C(tz),'formatted_string':S,G:E.isoformat()}
		except B as A:raise B(f"Invalid format string provided: {C(A)}")from A
		except J as A:raise N(f"Failed to format date: {C(A)}")from A
	def _handle_add(j,date_string,years,months,days,hours,minutes,seconds,tz):
		X=seconds;W=minutes;P=hours;M=months;L=years;K=date_string;H:0
		if not K:H=D.datetime.now(tz);Y=a
		else:
			try:k=j._handle_parse(K,tz);H=D.datetime.fromisoformat(k[G]);Y=K
			except B as Z:raise B(f"Failed to parse base date string for addition: {Z}")from Z
		l={Q:L,R:M,S:days,T:P,U:W,V:X};c=L or 0;d=M or 0;m=days or 0;o=P or 0;p=W or 0;q=X or 0
		try:
			E=H
			if c!=0 or d!=0:e=c*12+d;f=E.year+(E.month-1+e)//12;g=(E.month-1+e)%12+1;r=b.monthrange(f,g)[1];s=min(E.day,r);E=E.replace(year=f,month=g,day=s)
			t=D.timedelta(days=m,hours=o,minutes=p,seconds=q);E=E+t;return{F:n,O:Y,A1:{C:B for(C,B)in l.items()if B is not A},I:C(tz),'original_iso':H.isoformat(),'result_iso':E.isoformat(),'result_formatted':E.strftime(A0)}
		except OverflowError as h:raise B(f"Date calculation resulted in overflow: {C(h)}")from h
		except J as i:raise N(f"Failed to add time: {C(i)}")from i
	def _handle_subtract(K,date_string,years,months,days,hours,minutes,seconds,tz):
		J=seconds;I=minutes;H=hours;G=days;E=months;D=years;L=-D if D is not A else A;M=-E if E is not A else A;N=-G if G is not A else A;O=-H if H is not A else A;P=-I if I is not A else A;W=-J if J is not A else A
		try:C=K._handle_add(date_string,L,M,N,O,P,W,tz);C[F]=o;C['subtractions']={Q:D,R:E,S:G,T:H,U:I,V:J};C.pop(A1,A);return C
		except B as X:raise X
	def _handle_diff(K,date1_str,date2_str,tz):
		M=date2_str;L=date1_str
		if not L or not M:raise B("Both date1 and date2 strings are required for 'diff' operation.")
		try:Y=K._handle_parse(L,tz);Z=K._handle_parse(M,tz);O=D.datetime.fromisoformat(Y[G]);P=D.datetime.fromisoformat(Z[G]);A=P-O;a=A.total_seconds();E=A.days;H=a-E*86400;b=c(H//3600);d=c(H%3600//60);e=c(H%60);f=A.microseconds;g=E//365;i=E%365//30;return{F:p,'date1_iso':O.isoformat(),'date2_iso':P.isoformat(),I:C(tz),'total_seconds':A.total_seconds(),h:{Q:g,R:i,S:E,T:b,U:d,V:e,'microseconds':f,'total_days_td':A.days}}
		except B as W:raise B(f"Failed to parse date strings for diff operation: {W}")from W
		except J as X:raise N(f"Failed to calculate date difference: {C(X)}")from X
	def _handle_weekday(L,date_string,tz):
		E=date_string;A:0
		if not E:A=D.datetime.now(tz);H=a
		else:
			try:M=L._handle_parse(E,tz);A=D.datetime.fromisoformat(M[G]);H=E
			except B as J:raise B(f"Failed to parse date string for weekday check: {J}")from J
		N=A.weekday();P=A.strftime(Z);Q=A.isoweekday();return{F:K,O:H,I:C(tz),G:A.isoformat(),K:{A2:N,'number_iso':Q,'name':P}}
	def _handle_is_weekend(M,date_string,tz):
		E=date_string;A:0
		if not E:A=D.datetime.now(tz);H=a
		else:
			try:N=M._handle_parse(E,tz);A=D.datetime.fromisoformat(N[G]);H=E
			except B as J:raise B(f"Failed to parse date string for weekend check: {J}")from J
		L=A.weekday();P=L>=5;return{F:d,O:H,I:C(tz),G:A.isoformat(),d:P,K:{A2:L,'name':A.strftime(Z)}}
	def _handle_is_leap_year(I,year):
		E=year if year is not A else D.datetime.now().year
		if not 1<=E<=9999:raise B(A3)
		try:H=b.isleap(E);return{F:e,M:E,e:H}
		except J as G:raise N(f"Failed to check leap year: {C(G)}")from G
	def _handle_days_in_month(P,year,month):
		H=month;I=D.datetime.now();E=year if year is not A else I.year;G=H if H is not A else I.month
		if not 1<=E<=9999:raise B(A3)
		if not 1<=G<=12:raise B('Month must be between 1 and 12.')
		try:L=b.monthrange(E,G)[1];O=D.date(E,G,1).strftime(j);return{F:f,M:E,Y:G,i:O,f:L}
		except J as K:raise N(f"Failed to get days in month: {C(K)}")from K