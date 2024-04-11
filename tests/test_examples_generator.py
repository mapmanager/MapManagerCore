import os
import unittest
import nbformat


class TestPyNotebook(unittest.TestCase):

    def test_update_test_files(self):
        with open('./examples/example.ipynb', 'r') as f:
            nb = nbformat.read(f, as_version=4)

        current_file_path = os.path.abspath(__file__)
        test_file_path = os.path.join(os.path.dirname(
            current_file_path), 'test_gen_example_notebook.py')
        test_file = open(test_file_path, 'w')

        test_file.write("import matplotlib\n")
        test_file.write("import unittest\n")
        test_file.write("import os\n\n")
        test_file.write("class TestExamplesNotebook(unittest.TestCase):\n\n")

        code = ""

        for cell in nb.cells:
            if cell.cell_type == 'code':
                if cell.source.startswith('%'):
                    continue

                code += cell.source + '\n'

        test_file.write(f"    def test_notebook(self):\n")
        test_file.write("        wd = os.curdir\n")
        test_file.write("        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../examples/'))\n")
        test_file.write("        matplotlib.use('Agg')\n")
        test_file.write("        try:\n")
        test_file.write('            ' +
                        code.replace('\n', '\n            ') + '\n')
        test_file.write("        finally:\n")
        test_file.write("            os.chdir(wd)\n\n")

        test_file.write("if __name__ == '__main__':\n")
        test_file.write("    unittest.main()")
        test_file.close()


if __name__ == '__main__':
    unittest.main()
