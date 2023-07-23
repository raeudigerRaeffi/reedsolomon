# -*- coding: utf-8 -*-

# Copyright (c) 2012-2015 Tomer Filiba <tomerfiliba@gmail.com>
# Copyright (c) 2015 rotorgit
# Copyright (c) 2015-2023 Stephen Larroque <LRQ3000@gmail.com>
# Copyright (c) 2023 Raffael Scho <github: raeudigerRaeffi>

'''
Generalized Reed-Solomon Algorithm (Parallelized Reed-Solomon)
==============================================================

Implements a fast parallelized codec called the Generalized Reed-Solomon algorithm,
initially published in https://doi.org/10.1109/ITA.2013.6502932 and implemented
in https://github.com/raeudigerRaeffi/generalizedReedSolomon

DRAFT: the rest is a copy of reedsolo.py, it provides the main functions to reproduce the same API but the content of the functions need to be changed!

'''

# TODO IMPORTANT: try to keep the same convention for the ordering of polynomials inside lists throughout the code and functions (because for now there are a lot of list reversing in order to make it work, you never know the order of a polynomial, ie, if the first coefficient is the major degree or the constant term...).

import array
import numpy as np
import galois as gl
from . import basereedsolomon
import math


################### INIT and stuff ###################

try:  # pragma: no cover
    bytearray
    _bytearray = bytearray
except NameError:  # pragma: no cover
    def _bytearray(obj = 0, encoding = "latin-1"):  # pragma: no cover
        '''Simple pure-python bytearray replacement if not implemented'''
        # always use Latin-1 and not UTF8 because Latin-1 maps the first 256 characters to their bytevalue equivalents. UTF8 may mangle your data (particularly at vale 128)
        if isinstance(obj, str):
            obj = [ord(ch) for ch in obj.encode(encoding)]
        elif isinstance(obj, int):
            obj = [0] * obj
        return array.array("B", obj)

#try:  # pragma: no cover
    # compatibility with Python 2.7
#    xrange
#except NameError:  # pragma: no cover
    # compatibility with Python 3+
#    xrange = range



class Parallel_Reed_Solo(basereedsolomon.Base_Reed_Solomon):

    def __init__(self, field_size: int, message_length: int, payload_length: int, symbol_size: int, p_factor: int,
                 irr_poly=None, multi_processing=False, debug=True) -> None:
        #TODO Remove debug
        super().__init__(field_size, message_length, payload_length, symbol_size, multi_processing, irr_poly, debug,
                         p_factor)
        self.p = p_factor
        self.primitive = self.galois_field.primitive_element  # gl.primitive_root(1)

        self.p_unity = self.helper.get_nth_unity_root_of_field(p_factor)
        self.generator_set = []

        ## handles cases where either k or r are not divisible by p, implies by extension the n may not be divisible by p
        self.d = self.two_s % self.p  # remainder of r
        self.c = self.payload_length % self.p  # remainder of k
        self.num_of_padded_zeros = (self.p - self.c) if self.c != 0 else 0

        # adjust parameters to handle r not divisble by p
        if self.d != 0:
            raise ValueError("r % p  != 0 is so far not implemented")
            self.T = self.calculate_t_matrix()
            self.V_inverse = np.linalg.inv(self.calculate_v_matrix())
            self.tv_transformation_matrix = self.return_transformation_values_for_f(self.T, self.V_inverse)

        for index in range(0, p_factor):
            g_i = self.return_generalized_generator(index)
            self.generator_set.append(g_i)

        # divisibility of n, k and r should be checked

    def encode(self, message):
        if len(message) != self.payload_length:
            raise ValueError("Length not as specified")
        if self.c != 0:
            # add conceptual zeros
            message = ([0] * self.num_of_padded_zeros) + message


        msg = self.encode_classic(message)
        msg = msg[self.num_of_padded_zeros:len(
            msg)]  # + msg[self.payload_length + self.num_of_padded_zeros:len(msg)]
        return msg

    def decode(self, recieved_msg):
        if len(recieved_msg) != (self.message_length):
            raise ValueError("Length not as specified")
        if self.c != 0:
            # add conceptual zeros
            # recieved_msg =  recieved_msg[0:self.payload_length] +([0]*self.num_of_padded_zeros) + recieved_msg[self.payload_length:len(recieved_msg)]
            recieved_msg = ([0] * self.num_of_padded_zeros) + recieved_msg
        corrected_msg = self.decode_classic(recieved_msg)
        return self.return_info_symbols(corrected_msg)

    def encode_classic(self, message):
        # only used in case where self.d != 0
        coefficent_vector = []

        output_message = message

        message_matrix = self.input_arr_to_matrix(message)
        l = self.two_s // self.p
        # performs shift by x**l
        temp_message_matrix = []
        for row_id in range(0, message_matrix.shape[0]):
            msg_slice = np.copy(message_matrix[row_id])
            msg_slice = np.append(msg_slice, np.zeros(l, dtype=int))
            temp_message_matrix.append(msg_slice)
        message_matrix = np.array(temp_message_matrix)
        f_values = []

        fft = self.helper.fft_on_matrix_multi(message_matrix) if self.multi else self.helper.fft_on_matrix(
            message_matrix)
        for f_index in range(0, self.p):
            f_i = gl.Poly(fft[f_index], field=self.galois_field)
            f_values.append((f_i % self.generator_set[f_index]).coeffs.tolist())

        # calculates adjusted f
        if self.d != 0:
            # get coefficents of first d f values and remove them from vec
            for coeff_index in range(0, self.d):
                # gets coeff
                coeff_val = f_values[coeff_index][0]
                # removes coeff
                f_values[coeff_index] = f_values[coeff_index][1:len(f_values[coeff_index])]
                coefficent_vector.append(coeff_val)
            coefficent_vector = self.galois_field(coefficent_vector)
            gamma_adjustments = - np.dot(self.tv_transformation_matrix, coefficent_vector)  # - since f adj = f - gamma
            h_adjustments = np.dot(self.V_inverse, gamma_adjustments[0:self.d])  # used later to calc h adjusted

            for adjustment_f_index in range(self.d, self.p):
                f_i_to_be_adjusted = [int(gamma_adjustments[adjustment_f_index])] + ([0] * l)
                gamma_remainder = (gl.Poly(f_i_to_be_adjusted, field=self.galois_field) % self.generator_set[
                    adjustment_f_index])
                f_values[adjustment_f_index] = (
                            gl.Poly(f_values[adjustment_f_index], field=self.galois_field) + gamma_remainder).coeffs

        parity_values = []

        ifft = self.helper.ifft_on_matrix(f_values) if self.multi else self.helper.ifft_on_matrix(f_values)
        for h_index in range(0, self.p):
            h_i = ifft[h_index]
            # calculates adjusted h
            if self.d != 0:
                if h_index < self.d:
                    h_i = [int(h_adjustments[h_index])] + h_i

            parity_values.append(h_i)


        # for parity_value_index in range(0,l):
        #    for parity_array_index in range(0,self.p):
        #        output_message.append(int(parity_values[self.p-parity_array_index-1][parity_value_index]))
        output_message = self.append_parity_symbols(output_message, parity_values)


        return output_message

    def decode_classic(self, recieved_msg):

        syndromes = self.calc_syndrome(recieved_msg)

        if np.all(syndromes == self.galois_field(0)):
            return recieved_msg


        galois_lfsr = gl.berlekamp_massey(self.galois_field(syndromes), "galois")
        error_locator_polynominal = galois_lfsr.feedback_poly
        error_evaluator_polynominal = gl.Poly(galois_lfsr.state, field=self.galois_field)


        error_locations = self.modified_chien_search(error_locator_polynominal.coeffs[::-1])
        error_locations_index = [(self.message_length + self.num_of_padded_zeros) - x - 1 for x in
                                 error_locations]  # need to be len(recived msg -1 - location)

        error_magnitude = self.modified_forney(error_locator_polynominal, error_evaluator_polynominal, error_locations)
        correction_msg = self.galois_field([0] * len(recieved_msg))
        for index in range(0, len(error_locations_index)):
            correction_msg[error_locations_index[index]] = error_magnitude[index]

        return [int(symbol) for symbol in (self.galois_field(recieved_msg) + correction_msg)]

    def primitive_element_adjusted(self, index):
        factor_unity = index % self.p
        factor_primitive = divmod(index, self.p)[0]
        return self.p_unity ** factor_unity * self.primitive ** factor_primitive

    def return_generalized_generator(self, index):
        if index >= self.p:
            raise ValueError("Generator polynominals can only be generated from 0 to p-1")
            ## index is first value that qualifies i%self.p == index condition
        output_poly = gl.Poly([1, -self.primitive ** index], field=self.galois_field)
        ## start looping after index
        for i in range(index + 1, self.two_s):
            if i % self.p == index:
                output_poly *= gl.Poly([1, -self.primitive ** i], field=self.galois_field)
        return output_poly

    def input_arr_to_matrix(self, _arr):
        output = []
        p = self.p
        intermediate_arr = []
        for i in range(0, len(_arr)):
            intermediate_arr.append(_arr[i])
            if (i + 1) % p == 0:
                output.append(intermediate_arr[::-1])
                intermediate_arr = []
        return np.array(np.transpose(output))

    def calc_syndrome(self, recieved_codeword):

        output_syndromes = []
        m = len(recieved_codeword) // self.p
        # m = len(recieved_codeword) // self.p
        divided_codeword = []
        for l_index in range(0, self.p):
            syndrome = []
            for m_index in range(0, m):
                recieved_code_index = self.p * m_index + l_index
                syndrome.append(recieved_codeword[len(recieved_codeword) - recieved_code_index - 1])
            divided_codeword.append(syndrome[::-1])

        fourier_transformed_syndromes = []
        fft = self.helper.fft_on_matrix_multi(divided_codeword) if self.multi else self.helper.fft_on_matrix(
            divided_codeword)
        for F_i in range(0, self.p):
            fourier_transformed_syndromes.append(fft[F_i])

        for s_i in range(0, self.two_s):
            primitive_root = self.primitive ** s_i
            fourier_poly = gl.Poly(fourier_transformed_syndromes[s_i % self.p], field=self.galois_field)
            output_syndromes.append(fourier_poly(primitive_root))

        return output_syndromes

    # modified version for parallelization
    def modified_chien_search(self, error_locator_poly):
        error_locations = []

        m = (self.message_length + self.num_of_padded_zeros) // self.p
        for i in range(0, m):
            modified_a_values = []
            for a_index in range(0, self.p):
                a_i = self.galois_field(0)
                for j_ai_index in range(0, len(error_locator_poly)):
                    if j_ai_index % self.p == a_index:
                        a_i = a_i + error_locator_poly[j_ai_index] * self.primitive ** -(j_ai_index * i)
                modified_a_values.append(a_i)
            ifft = np.fft.ifft(self.galois_field(modified_a_values))
            for w_index in range(0, self.p):
                if ifft[w_index] == self.galois_field(0):
                    error_locations.append(self.p * i + w_index)
        return error_locations

    def modified_forney(self, error_location_poly, error_evaluator_poly, error_loc):
        error_magnitudes = []
        for r in error_loc:
            a_i = self.primitive_element_adjusted(r)

            error_magnitudes.append(
                (a_i * error_evaluator_poly(a_i ** -1)) / error_location_poly.derivative()(a_i ** -1))
        return error_magnitudes

    def return_info_symbols(self, msg):
        return msg[self.num_of_padded_zeros:self.num_of_padded_zeros + self.payload_length]

    def calculate_t_matrix(self):
        matrix = []
        for i in range(self.d, self.p):
            row = [1]
            for j in range(1, self.d):
                row.append(self.p_unity ** (j * i))
            matrix.append(row)
        return self.galois_field(np.matrix(matrix))

    def calculate_v_matrix(self):
        matrix = []
        matrix.append([1] * self.d)
        for i in range(1, self.d):
            row = [1]
            for j in range(1, self.d):
                row.append(self.p_unity ** (j * i))
            matrix.append(row)
        return self.galois_field(np.matrix(matrix))

    def return_transformation_values_for_f(self, T, V_inverse):
        output_matrix = []
        t_vinverse_matrices = np.dot(T, V_inverse)
        # generate identity matrix
        for i in range(0, self.d):
            identity_row = [0] * self.d
            identity_row[i] = 1
            output_matrix.append(identity_row)
        for row in t_vinverse_matrices:
            output_matrix.append(row)
        return self.galois_field(np.matrix(output_matrix))

    def append_parity_symbols(self, message, parity_symbols):
        output_msg = message
        output_symbols = []
        parity_symbols = parity_symbols
        while len(parity_symbols) != 0:
            for parity_arry in parity_symbols:
                if len(parity_arry) == 0:
                    # since we are back to forth the first element is always the first to be empty
                    parity_symbols.pop(0)
                    break
                parity_symbol = parity_arry.pop(len(parity_arry) - 1)
                output_symbols.append(parity_symbol)

        return output_msg + output_symbols[::-1]

