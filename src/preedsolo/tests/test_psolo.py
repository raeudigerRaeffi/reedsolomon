import unittest
import numpy as np
from ..preedsolo import Parallel_Reed_Solo



class TestReedSolomon(unittest.TestCase):


    def test_base_fortyseven(self):
        ##purpose is to test higher degrees
        info_symbols = [
            [20, 5],
            [35, 10],
            [5, 10],
            [35, 20],
            [35, 10],
            [3, 10],
            [35, 10],
            [35, 2],
            [5, 10],
            [35, 10]
        ]

        test_normal_rs = Parallel_Reed_Solo(field_size=47,
                                            message_length=14,
                                            payload_length=10,
                                            symbol_size=2,
                                            p_factor=2)
        # convert to galois elements
        symbols = test_normal_rs.convert_to_symbol_array(info_symbols)
        symbols_encoded = test_normal_rs.encode(symbols)
        symbols_encoded[0] = 1
        # convert Galois Elements back to coefficent representation
        symbols_decoded = test_normal_rs.decode(symbols_encoded)
        conv_back = test_normal_rs.symbol_array_to_array(symbols_decoded)

        self.assertTrue(info_symbols == conv_back)


    def test_base_thirtyone(self):
        org_info = [30, 20, 20, 9, 25, 1, 7, 2, 0, 18, 17, 22, 11, 0, 0]
        org_compare = np.copy(org_info)
        test_normal_rs = Parallel_Reed_Solo(field_size=31,
                                            message_length=27,
                                            payload_length=15,
                                            symbol_size=1,
                                            p_factor=3)
        # encode message
        normal_msg = test_normal_rs.encode(org_info)
        # corrupt encode message
        normal_msg[6] = 10
        normal_msg[9] = 12
        normal_msg[10] = 0
        normal_msg[16] = 4
        normal_msg[23] = 10

        # fix encoded message
        dec_test_normal = test_normal_rs.decode(normal_msg)

        self.assertTrue((org_compare == dec_test_normal).all())