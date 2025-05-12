E='CRITICAL'
D='HIGH'
C='NORMAL'
B='LOW'
from enum import Enum as A
class F(str,A):
	LOW=B;NORMAL=C;HIGH=D;CRITICAL=E
	def as_int(A):return{B:4,C:3,D:2,E:1}[A.value]
class G(str,A):PENDING='pending';RUNNING='running';COMPLETED='completed';FAILED='failed';CANCELED='canceled'