import os
import xml.etree.ElementTree as ET
from neurobox.io.load_xml import load_xml

def file_exists(file_name):
    return os.path.exists(file_name)


def load_par(file_name, spec_info=1):
    par = {}

    # Check if file exists
    if not file_exists(file_name):
        # file_name = resolve_path(file_name, 0)  # Add your resolve_path logic here
        pass

    if '.par' in file_name:
        file_base = file_name.split('.par')[0]
    elif '.xml' in file_name:
        file_base = file_name.split('.xml')[0]
    else:
        file_base = file_name

    if file_exists(f"{file_base}.xml"):
        par = load_xml(f"{file_base}.xml")

    elif file_exists(f"{file_base}.par"):
        with open(f"{file_base}.par", 'r') as fp:
            par['file_name'] = file_base

            # Read n_channels and n_bits
            line = fp.readline().strip()
            a = list(map(int, line.split()))
            par['n_channels'] = a[0]
            par['n_bits'] = a[1]

            # Read sample_time and hi_pass_freq
            line = fp.readline().strip()
            a = list(map(float, line.split()))
            par['sample_time'] = a[0]
            par['hi_pass_freq'] = a[1]

            # Read n_elec_gps
            line = fp.readline().strip()
            if not line:
                return par
            a = int(line)
            par['n_elec_gps'] = a

            # Read elec_gp
            par['elec_gp'] = []
            for i in range(par['n_elec_gps']):
                line = fp.readline().strip()
                a = list(map(int, line.split()))
                par['elec_gp'].append(a[1:])
    else:
        raise FileNotFoundError('Par or Xml file does not exist!')

    return par
