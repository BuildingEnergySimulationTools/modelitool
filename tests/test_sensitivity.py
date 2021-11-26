from modelitool.sensitivity import modelitool_to_salib_problem


class TestSensitivity:
    def test_modelitool_to_salib_problem(self):
        sa_problem = {
            'num_vars': 5,
            'names': [
                'f_stegos.k',
                'R_stegos.R',
                'alpha_alu',
                'R_al21',
                'C_al',
            ],
            'bounds': [
                [0, 0.2],
                [0.023 - 0.023 * 0.3, 0.023 + 0.023 * 0.3],
                [0.2 - 0.2 * 0.3, 0.2 + 0.2 * 0.3],
                [0.00005 - 0.00005 * 0.3, 0.00005 + 0.00005 * 0.3],
                [2700 - 2700 * 0.3, 2700 + 2700 * 0.3]
            ]
        }

        as_config = {
            'f_stegos.k': [0, 0.2],
            'R_stegos.R': [0.023 - 0.023 * 0.3, 0.023 + 0.023 * 0.3],
            'alpha_alu': [0.2 - 0.2 * 0.3, 0.2 + 0.2 * 0.3],
            'R_al21': [0.00005 - 0.00005 * 0.3, 0.00005 + 0.00005 * 0.3],
            'C_al': [2700 - 2700 * 0.3, 2700 + 2700 * 0.3],
        }

        assert modelitool_to_salib_problem(as_config) == sa_problem
