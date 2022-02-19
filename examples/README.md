# Developing MAICoS modules

## Custom modules

To add your custom module to the maicos library follow these steps

1.  Create a `.maicos` folder in your home directory.
2.  Copy the `analysis_example.py` and the `maicos_custom_modules.py` to the 
    `.maicos` folder
3.  Add your analysis code to the `analysis_example.py`
4.  OPTIONAL: Choose your own name for your module. Note that you also have
    adjust the names in the `maicos_custom_modules.py` properly

## Add modules to the core library (MAICoS developers)

1.  Choose an unique `<name>` for your module and create a file in the modules directory. You can use the `analyse_example` as template if you like.
2.  Add `<name>` to the `__all__` list in `maicos/__init__.py` and add
    `from .<name> import *` to `maicos/modules/__init__.py`.
3.  OPTIONAL: Add bash completion commands to "mmaniocsshare/maicos_completion.bash".
