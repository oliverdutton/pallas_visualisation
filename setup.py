from setuptools import setup, find_packages

setup(
	name="pallas_visualisation",
	version="1.0.0",
	packages=find_packages(),
	description="A visualisation tool for Pallas",
	author="oliverdutton",
	url="https://github.com/oliverdutton/pallas_visualisation",
	install_requires=[
		"setuptools",
		"jax==0.4.30",
		"gradio",
		"chalk-diagrams @ git+https://github.com/chalk-diagrams/chalk.git",
	],
)
