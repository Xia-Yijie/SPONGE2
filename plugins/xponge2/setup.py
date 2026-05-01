from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = super().get_tag()
        if python.startswith("cp"):
            python = "cp310"
            abi = "abi3"
        return python, abi, plat


setup(cmdclass={"bdist_wheel": bdist_wheel})
