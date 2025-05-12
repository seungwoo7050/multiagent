C=print
import sys
from functools import lru_cache as D
import logging as E
try:from src.schemas.config import AppSettings as A
except ImportError:C('Warning: Could not import AppSettings from src.schemas.config. Using fallback BaseSettings.',file=sys.stderr);from pydantic_settings import BaseSettings as F;A=F
B=E.getLogger(__name__)
@D()
def G():
	try:E=A();B.debug('Loaded application settings.');return E
	except Exception as F:D=f"Failed to load settings: {F}";B.error(D,exc_info=True);C(f"FATAL: {D}",file=sys.stderr);raise
H=G()