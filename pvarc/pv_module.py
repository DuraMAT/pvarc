import numpy as np
from math import pi

# Suppose this is your multilayer function you either implement or import:
from pvarc.fresnel import multilayer

from pvarc import thin_film_transmittance, thick_film_absorbance, single_interface_transmittance, double_film, \
    path_length_enhancement


from scipy.interpolate import interp2d

from pvarc.materials import (
    refractive_index_porous_silica,
    refractive_index_glass,
    refractive_index_eva,
    refractive_index_SiN,
    refractive_index_silicon,
    refractive_index_amorphous_silicon,
    refractive_index_ito
)


class PVStack:
    def __init__(self,
                 cell_type='PERC',
                 index_glass_coating=1.2,
                 index_glass=1.5,
                 index_encapsulant=1.5,
                 index_cell_coating_1=3.5,
                 index_cell_coating_2=3.5,
                 index_cell=3.5,
                 porous_silica_porosity=0.28,
                 thickness_glass_coating=140,
                 thickness_glass=2e-3,
                 thickness_encapsulant=0.45e-3,
                 thickness_cell_coating_1=70,
                 thickness_cell_coating_2=0,
                 cell_path_length_enhancement_width=50,
                 cell_path_length_enhancement_height=10,
                 cell_path_length_enhancement_cutoff=1100,
                 collection_probability_cell_coating=0.25,
                 front_glass_angle_correction_factor = 0.992,
                 thickness_cell=300e3,
                 wavelength=1000,
                 wavelength_list=None,
                 aoi_list=None,
                 spectrum=None,
                 light_redirection_factor=0,
                 cell_arc_physical_improvement_factor=2,
                 aoi=8.0,
                 polarization='mixed',
                 index_air=1.0003):

        self.index_glass_coating = index_glass_coating
        self.index_glass = index_glass
        self.index_encapsulant = index_encapsulant
        self.index_cell_coating_1 = index_cell_coating_1
        self.index_cell_coating_2 = index_cell_coating_2
        self.index_cell = index_cell
        self.porous_silica_porosity = porous_silica_porosity
        self.thickness_glass_coating = thickness_glass_coating
        self.thickness_glass = thickness_glass
        self.thickness_encapsulant = thickness_encapsulant
        self.thickness_cell_coating_1 = thickness_cell_coating_1
        self.thickness_cell_coating_2 = thickness_cell_coating_2

        self.cell_path_length_enhancement_width = cell_path_length_enhancement_width
        self.cell_path_length_enhancement_height = cell_path_length_enhancement_height
        self.cell_path_length_enhancement_cutoff = cell_path_length_enhancement_cutoff
        self.collection_probability_cell_coating = collection_probability_cell_coating

        self.thickness_cell = thickness_cell
        self.wavelength = wavelength
        self.wavelength_list = wavelength_list
        self.light_redirection_factor = light_redirection_factor
        self.cell_arc_physical_improvement_factor = cell_arc_physical_improvement_factor
        self.aoi = aoi
        self.aoi_list = aoi_list
        self.polarization = polarization
        self.index_air = index_air

        self.front_glass_angle_correction_factor = front_glass_angle_correction_factor

        # spectrum is an array with shape (len(aoi_series), len(wavelength_series))
        # self.wavelength_series = None
        self.spectrum_series = spectrum
        self.aoi_series = None

        self.create_wavelength_aoi_grid()

        self.load_default_module_parameters(cell_type)

        self.grid_result = None

    def load_default_module_parameters(self, cell_type='PERC'):

        wavelength = self.wavelength
        if cell_type.upper() == 'PERC' or cell_type.upper() == 'TOPCON':
            self.index_glass_coating = refractive_index_porous_silica(wavelength, porosity=self.porous_silica_porosity)
            self.index_glass = refractive_index_glass(wavelength)

            self.index_encapsulant = refractive_index_eva(wavelength)
            self.index_cell_coating_1 = refractive_index_SiN(wavelength)
            self.index_cell_coating_2 = refractive_index_SiN(wavelength)
            self.index_cell = refractive_index_silicon(wavelength)
            self.index_air = 1.0003

            # self.thickness_cell = 300e3
            self.thickness_cell = 130e3

            self.cell_arc_physical_improvement_factor = 20
            self.thickness_cell_coating_1 = 70
            self.thickness_cell_coating_2 = 1
            self.thickness_encapsulant = 0.45e-3 * 1e9
            self.thickness_glass = 2e-3 * 1e9
            # self.thickness_glass_coating = 128
            self.thickness_glass_coating = 125


        elif cell_type == 'HJT':
            self.index_glass_coating = refractive_index_porous_silica(wavelength, porosity=0.28)
            self.index_glass = refractive_index_glass(wavelength)

            self.index_encapsulant = refractive_index_eva(wavelength)

            self.index_cell_coating_1 = refractive_index_ito(wavelength)
            self.index_cell_coating_2 = refractive_index_amorphous_silicon(wavelength)
            self.index_cell = refractive_index_silicon(wavelength)
            self.index_air = 1.0003

            # self.thickness_cell = 300e3
            self.thickness_cell = 130e3
            self.cell_arc_physical_improvement_factor = 20
            self.thickness_cell_coating_1 = 70
            self.thickness_cell_coating_2 = 9
            self.thickness_encapsulant = 0.45e-3 * 1e9
            self.thickness_glass = 2e-3 * 1e9
            self.thickness_glass_coating = 125

            # self.cell_path_length_enhancement_width = 30
            # self.cell_path_length_enhancement_height = 30
            # self.cell_path_length_enhancement_cutoff = 1070

        elif cell_type == 'CdTe':
            pass
        else:
            raise ValueError('Invalid cell type')

    def create_wavelength_aoi_grid(self):
        # if wavelength_list is None:
        #     wavelength_list = np.arange(300, 1201, 1)
        # if aoi_list is None:
        #     # aoi_list = np.arange(0, 90.1, 0.1)
        #     aoi_list = np.arange(0, 91, 1)
        if self.wavelength_list is None:
            self.wavelength_list = np.arange(300, 1201, 1)

        if self.aoi_list is None:
            self.aoi_list = np.arange(0, 91, 1)

        # self.wavelength_list = wavelength_list
        # self.aoi_list = aoi_list

        self.wavelength = np.repeat(np.atleast_2d(self.wavelength_list).T, len(self.aoi_list), axis=1)
        self.aoi = np.repeat(np.atleast_2d(self.aoi_list), len(self.wavelength_list), axis=0)
        self.absorbance = np.zeros((len(self.aoi_list), len(self.wavelength_list)))

    def calculate_angles(self):
        theta0 = self.aoi * pi / 180
        theta1 = np.arcsin(np.real(self.index_air) / np.real(self.index_glass_coating) * np.sin(theta0))
        theta2 = np.arcsin(np.real(self.index_air) / np.real(self.index_glass) * np.sin(theta0))
        theta3 = np.arcsin(np.real(self.index_air) / np.real(self.index_encapsulant) * np.sin(theta0))
        theta4 = np.arcsin(np.real(self.index_air) / np.real(self.index_cell_coating_1) * np.sin(theta0))
        theta5 = np.arcsin(np.real(self.index_air) / np.real(self.index_cell_coating_1) * np.sin(theta0))
        theta6 = np.arcsin(np.real(self.index_air) / np.real(self.index_cell) * np.sin(theta0))

        return theta0, theta1, theta2, theta3, theta4, theta5, theta6

    def calculate_absorbance(self):
        theta0, theta1, theta2, theta3, theta4, theta5, theta6 = self.calculate_angles()

        # Give index of refraction values better names
        n0, n1, n2, n3, n4, n5, n6 = (self.index_air, self.index_glass_coating, self.index_glass,
                                   self.index_encapsulant, self.index_cell_coating_1, self.index_cell_coating_2,
                                      self.index_cell)

        # def effective_aoi(aoi, delta_angle_rad):
        #     # weighted average of the angle of incidence
        #     aoi_plus = aoi + delta_angle_rad
        #     aoi_minus = aoi - delta_angle_rad
        #
        #     weight_plus = np.cos(aoi_plus)
        #     weight_minus = np.cos(aoi_minus)
        #
        #     effective_aoi = (weight_plus * aoi_plus + weight_minus * aoi_minus) / (weight_plus + weight_minus)
        #
        #     return effective_aoi
        #
        # def effective_aoi(theta, delta_angle_rad):
        #     # weighted average of the angle of incidence
        #     aoi_plus = theta + delta_angle_rad
        #     aoi_minus = theta - delta_angle_rad
        #
        #     def weight_function(aoi):
        #         return np.cos(aoi) * (np.cos(aoi) ** 2 + 1) / 2
        #
        #     weight_plus = weight_function(aoi_plus)
        #     weight_minus = weight_function(aoi_minus)
        #
        #
        #     effective_aoi = (weight_plus * aoi_plus + weight_minus * aoi_minus) / (weight_plus + weight_minus)
        #
        #     return effective_aoi

        theta0_new = theta0.copy()
        theta0_new = self.front_glass_angle_correction_factor * theta0
        # theta0_new[theta0_new > 89 * np.pi / 180] = theta0
        theta0_new[theta0 > 89 * np.pi / 180] = theta0[theta0 > 89 * np.pi / 180]

        theta0 = theta0_new
        # theta0 = effective_aoi(theta0, 3*np.pi/180)

        # Transmission from air through glass coating into the glass
        T1 = thin_film_transmittance(n1, n2, self.thickness_glass_coating, self.wavelength, theta0 * 180 / np.pi, self.polarization, n0)
        R1 = 1 - T1

        # weight by
        # front_glass_texture_angle = 7
        # T1_plus = thin_film_transmittance(n1, n2, self.thickness_glass_coating, self.wavelength,
        #                                   theta0 * 180 / np.pi + front_glass_texture_angle, self.polarization, n0)
        # T1_minus = thin_film_transmittance(n1, n2, self.thickness_glass_coating, self.wavelength,
        #                                      theta0 * 180 / np.pi - front_glass_texture_angle, self.polarization, n0)
        # weighting_T1_plus = np.cos(theta0 + front_glass_texture_angle / 180 * np.pi)
        # weighting_T1_minus = np.cos(theta0 - front_glass_texture_angle / 180 * np.pi)
        # T1 = (T1_plus * weighting_T1_plus + T1_minus * weighting_T1_minus) / (weighting_T1_plus + weighting_T1_minus)


        # Absorbance in the glass
        A2 = thick_film_absorbance(n2, self.thickness_glass, self.wavelength, theta2 * 180 / pi)
        # Transmission from glass into encapsulant, non-coherent.
        T2 = single_interface_transmittance(n2, n3, theta2 * 180 / pi, self.polarization)
        # Absorbance in the encapsulant
        A3 = thick_film_absorbance(n3, self.thickness_encapsulant, self.wavelength, theta3 * 180 / pi)

        # A bit of light is reflected back from the air-glass interface.
        R1_extra = (1 - T1) * self.light_redirection_factor
        # T4 = thin_film_transmittance(n4, n5, self.thickness_cell_coating_1, self.wavelength, theta3 * 180 / pi, self.polarization, n3)


        R4, T4, A4 = double_film(n3, n4, n5, n6, self.thickness_cell_coating_1, self.thickness_cell_coating_2,
                                 self.wavelength, theta3 * 180 / pi, self.polarization)

        # A4 = thick_film_absorbance(n4, self.thickness_cell_coating, self.wavelength, theta4 * 180 / pi)
        # T5 =
        # print('fix this absorbance!')

        T4 = 1 - (1 - T4) / self.cell_arc_physical_improvement_factor
        R4 = (1 - T4)

        cell_path_length_enhancement = path_length_enhancement(self.wavelength,width=self.cell_path_length_enhancement_width,
                                                               height=self.cell_path_length_enhancement_height,
                                                               cutoff=self.cell_path_length_enhancement_cutoff)

        # cell_path_length_enhancement = 1

        # cell_path_length_enhancement = 1


        angle_in_cell = theta6 * 180 / pi

        angle_in_cell = angle_in_cell * 1.8
    #
        # angle_in_cell = np.clip(angle_in_cell, 0, 90)


        A6 = thick_film_absorbance(n6, self.thickness_cell, self.wavelength, angle_in_cell,
                                   path_length_enhancement=cell_path_length_enhancement)


        # Aggregate results
        # eqe = np.real((T1 + R1) * (1 - A2) * T2 * (1 - A3) * ( T4 * (1-A4) * A6 + A4 * self.collection_probability_cell_coating ))
        eqe = np.real((T1 + R1_extra) * (1 - A2) * T2 * (1 - A3) * ( T4 * (1-A4) * A6 + A4 * self.collection_probability_cell_coating ))
        isc = eqe * self.wavelength / 1239.8


        self.grid_result = {
            'wavelength [nm]': self.wavelength,
            'aoi [degrees]': self.aoi,
            'EQE': eqe,
            'SR': isc,
            'Reflectance Glass ARC': np.real(R1),
            'Light Entering Cell': np.real(T1 * (1 - A2) * T2 * (1 - A3)),
            'Transmittance Glass ARC to Glass': np.atleast_1d(T1),
            'Absorbance Glass': np.atleast_1d(A2),
            'Transmittance Glass to Encapsulant': np.atleast_1d(T2),
            'Absorbance Encapsulant': np.atleast_1d(A3),
            'Absorbance Cell Coating': np.atleast_1d(A4),
            'Reflectance Cell Coating': np.atleast_1d(R4),
            'Transmittance Through Cell ARC to Cell': np.atleast_1d(T4)*(1 - np.atleast_1d(A4)),
            'Absorbance Cell': np.atleast_1d(A6),
        }

        return self.grid_result


    def calculate_series(self,extra_calculation_key=None):

        dwavelength = np.diff(self.wavelength_list)
        dwavelength = np.append(dwavelength, dwavelength[-1])

        sr_grid = self.grid_result['SR'].transpose()

        # Integrate on wavelength
        sr_series = np.dot(sr_grid, self.spectrum_series * np.atleast_2d(dwavelength).T)

        aoi_idx = [np.argmin(np.abs(self.aoi_list - aoi)) for aoi in self.aoi_series]
        # Interpolate on angle of incidence
        response_at_aoi = np.array([sr_series[aoi_idx[i],i] for i in range(len(self.aoi_series))])

        response_at_aoi[response_at_aoi<0] = 0
        ret = {'Photocurrent': response_at_aoi}
        if extra_calculation_key is not None:
            for key in extra_calculation_key:
                data_grid = self.grid_result[key].transpose()
                # Integrate on wavelength
                data_series = np.dot(data_grid, self.spectrum_series * np.atleast_2d(dwavelength).T)
                # Interpolate on angle of incidence
                data_at_aoi = np.array([data_series[aoi_idx[i],i] for i in range(len(self.aoi_series))])

                ret[key] = data_at_aoi




        return ret






if __name__ == '__main__':
    # pass
    pvs = PVStack(cell_type='HJT')
    ret = pvs.calculate_absorbance()
    print(ret.keys())

    pvs.aoi_series = np.linspace(0,0,5)
    pvs.spectrum_series = np.repeat( np.atleast_2d(np.ones_like(pvs.wavelength_list)), len(pvs.aoi_series), axis=0).transpose()

    pvs.spectrum_series.shape

    result = pvs.calculate_series()
    result['Photocurrent']

