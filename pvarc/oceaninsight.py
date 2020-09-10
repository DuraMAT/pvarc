import numpy as np
import pandas as pd

def read_oceanview_file(filename):
    data = {}
    data['filename'] = filename

    # Read header.
    with open(filename, 'r') as file:

        # is_header = True
        for j in range(100):

            line = file.readline()

            if line[:5] == '>>>>>':
                break

            if line.find(':') >= 0:
                line_parts = line.split(': ')
                data[line_parts[0]] = line_parts[1].replace('\n', '')

    skip_rows = j + 1

    data['Number of Pixels in Spectrum'] = int(data['Number of Pixels in Spectrum'])
    data['Integration Time (sec)'] = float(data['Integration Time (sec)'])
    data['Scans to average'] = int(data['Scans to average'])

    df = pd.read_csv(filename,
                     delimiter='\t',
                     skiprows=skip_rows,
                     header=None,
                     names=['wavelength', 'value']
                     )

    # Sometimes there are repeated pixels.
    data['wavelength'] = np.array(df['wavelength'])[
                         :data['Number of Pixels in Spectrum']]
    data['value'] = np.array(df['value'])[
                    :data['Number of Pixels in Spectrum']]
    return data


def read_oceanview_image_set(filenames):
    if len(filenames) == 0:
        raise Exception('no filenames given.')

    for k in range(len(filenames)):
        ref = read_oceanview_file(filenames[k])
        if k == 0:
            ref0 = ref.copy()
            value = np.zeros(
                (len(filenames), len(ref['wavelength'])))
            wavelength = ref['wavelength']

        value[k, :] = ref['value']

    value_mean = np.mean(value, axis=0)

    ref0['value'] = value
    ref0['value_mean'] = value_mean

    return ref0