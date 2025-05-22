'''
Commenting these lines, you would access the "normalize" function inside the "image_processing" module like this:
>>> from pymif import image_preprocessing as imgpre
>>> imgpre.normalize
Or
>>> import pymif.image_preprocessing as pmpre
>>> pmpre.normalize

These lines would make the functions inside the modules also available directly from nicola_pckg.
Because they make the imports of, e.g., image_preprocessing from the __init__ inside.
E.g.
>>> import pymif as pm
>>> pm.normalize
'''
# from .imagej_funs import *
# from .image_preprocessing import *
# from .cell_detection import *
# from .animations import *
# from .io import *

'''
with these lines only, one can access the functions with:
>>> import pymif as pm
>>> pm.image_preprocessing.normalize
or
>>> import pymif.image_preprocessing as pmpre
>>> pmpre.normalize
'''
# from . import image_preprocessing
# from . import imagej_funs
# from . import io
# from . import cell_detection
# from . import animations

# from . import luxendo
# from . import pe_opera

__version__ = "0.0.1"
__common_alias__ = "pm"
