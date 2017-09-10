from unittest import TestCase

from tensorflow import Tensor

from sample_models import multi_layer_cnn_output_length, cnn_output_length


class TestMulti_layer_cnn_output_length(TestCase):
    def test_multi_layer_cnn_output_length(self):
        params = {
            "filter_size": 256,
            "border_mode": "valid",
            "stride": 1,
            "dilation": 1
        }

        cnn_shapes = [params]
        # input_length =
        actual = multi_layer_cnn_output_length(100, cnn_shapes)
        expected = cnn_output_length(100, params["filter_size"],
                                     params["border_mode"], params["stride"],
                                     params["dilation"])

        result = cnn_output_length(0, params["filter_size"],
                                     params["border_mode"], params["stride"],
                                     params["dilation"])
        self.assertEqual(actual, expected)
