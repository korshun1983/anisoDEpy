import json
from pathlib import Path
import numpy as np
def load_model(json_path):
    """Read user JSON -> dict with numpy arrays."""
    with open(json_path, 'r') as f:
        d = json.load(f)
        m = d['Model']

    for key in ('DomainRx', 'DomainRy', 'DomainTheta', 'DomainEcc',
    'DomainEccAngle', 'DomainParam'):
        if key in m:
            m[key] = np.asarray(m[key])
        if 'f_array_range' in m:
            r = m['f_array_range']
        m['f_array'] = np.arange(r['start'], r['end'] + r['step']/2, r['step'])

    return d