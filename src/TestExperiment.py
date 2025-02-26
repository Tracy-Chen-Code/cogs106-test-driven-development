import unittest
from SignalDetection import SignalDetection
from Experiment import Experiment
#with the help of chatgpt and groupmates
class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.exp = Experiment()
        self.sdt1 = SignalDetection(10, 5, 3, 12)
        self.sdt2 = SignalDetection(20, 10, 5, 15)
        self.sdt3 = SignalDetection(15, 5, 2, 18)
        self.exp.add_condition(self.sdt1, label="Condition A")
        self.exp.add_condition(self.sdt2, label="Condition B")
        self.exp.add_condition(self.sdt3, label="Condition C")

    def test_add_condition(self):
        self.assertEqual(len(self.exp.conditions), 3)
        self.assertEqual(self.exp.conditions[0][1], "Condition A")

    def test_sorted_roc_points(self):
        fa_rates, hit_rates = self.exp.sorted_roc_points()
        sorted_pairs = sorted(zip(fa_rates, hit_rates))  
        fa_sorted, hit_sorted = zip(*sorted_pairs)  

        self.assertEqual(fa_rates, list(fa_sorted))
        self.assertEqual(hit_rates, list(hit_sorted))

    def test_compute_auc(self):
        auc = self.exp.compute_auc()
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_empty_experiment(self):
        empty_exp = Experiment()
        with self.assertRaises(ValueError):
            empty_exp.sorted_roc_points()
        with self.assertRaises(ValueError):
            empty_exp.compute_auc()

    def test_auc_half_and_one(self):
        exp_half = Experiment()
        exp_half.add_condition(SignalDetection(0, 0, 0, 0))  # (0,0)
        exp_half.add_condition(SignalDetection(1, 0, 1, 0))  # (1,1)
        
        fa_rates, hit_rates = exp_half.sorted_roc_points()
        print("False Alarm Rates:", fa_rates)
        print("Hit Rates:", hit_rates)
        print("Computed AUC:", exp_half.compute_auc())

        self.assertAlmostEqual(exp_half.compute_auc(), 0.5)  # Expected AUC = 0.5

        exp_one = Experiment()
        exp_one.add_condition(SignalDetection(0, 0, 0, 10))
        exp_one.add_condition(SignalDetection(10, 0, 10, 0))
        self.assertAlmostEqual(exp_one.compute_auc(), 0.5)

        exp_one_alt = Experiment()
        exp_one_alt.add_condition(SignalDetection(0, 0, 0, 10))
        exp_one_alt.add_condition(SignalDetection(10, 0, 0, 10))
        exp_one_alt.add_condition(SignalDetection(10, 0, 10, 0))
        self.assertAlmostEqual(exp_one_alt.compute_auc(), 1.0)

    def test_plot_roc_curve(self):
        try:
            self.exp.plot_roc_curve()
        except Exception as e:
            self.fail(f"plot_roc_curve() raised an exception: {e}")

    def test_hit_rate_calculation(self):
        sdt = SignalDetection(10, 5, 3, 12)
        self.assertAlmostEqual(sdt.hit_rate(), 10 / (10 + 5))

    def test_false_alarm_rate_calculation(self):
        sdt = SignalDetection(10, 5, 3, 12)
        self.assertAlmostEqual(sdt.false_alarm_rate(), 3 / (3 + 12))

if __name__ == "__main__":
    unittest.main()