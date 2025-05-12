import logging as A
E=A.getLogger('config.bootstrap')
C=A.StreamHandler()
C.setFormatter(A.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
E.addHandler(C)
C.setLevel(A.INFO)
B=None
D=None
def F():
	global B,D
	try:from src.config.settings import get_settings as A;B=A();from src.utils.logger import get_logger as C,setup_logging as F;F();D=C(__name__);D.info(f"Initializing {B.APP_NAME} v{B.APP_VERSION}...");return True
	except Exception as G:E.error(f"Failed to initialize configuration: {G}",exc_info=True);return False