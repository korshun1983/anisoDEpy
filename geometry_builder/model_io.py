import json
from pathlib import Path
import numpy as np


def load_model(json_path):
    """Read user JSON -> dict with numpy arrays."""
    with open(json_path, 'r') as f:
        d = json.load(f)
        m = d['Model']

    # Обработка обычных ключей с однородными данными
    for key in ('DomainRx', 'DomainRy', 'DomainTheta', 'DomainEcc', 'DomainEccAngle'):
        if key in m:
            m[key] = np.asarray(m[key], dtype=float)

    # Особые случаи обработки
    if 'DomainParam' in m:
        # DomainParam может содержать списки разной длины - сохраняем как список массивов
        m['DomainParam'] = [np.asarray(arr, dtype=float) for arr in m['DomainParam']]

    # Создание f_array
    if 'f_array_range' in m:
        r = m['f_array_range']
        m['f_array'] = np.arange(r['start'], r['end'] + r['step'] / 2, r['step'])

    return d