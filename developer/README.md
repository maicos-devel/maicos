# Developing for mdtools

Contribution via pull requests are always welcome. Before submitting a pull
request please open an issue

# Code formatting

We use [yapf](https//github.com/google/yapf) using the
`google` formatting style for our code. You can style
your code from the command line or using an extension for your favorite editor.

The easiest use is to install the git hook module, which will automatically format your
code before commiting. To install it just run the `enable_githooks.sh` from
the command line.

Currently we only format python files.

# Writing your own Analysis module

Example code for an analysis module can be found in the [example folder](examples/). To deploy
the script do the following steps.

1.  Copy it to the `mdtsools/ana` folder and add your code. To the following methods
    -   `__init__` Save the arguments to the objects namespace. Do not add any logic here,
                    instead use the `_prepare` method!
    -   `_prepare` Set up variables needed for your analysis.
    -   `_single_frame` The calcualtions run in every frame.
    -   `_calculate_results` Calculate your results based on the calculation in every frame.
                              Save them to the objects `results` dictionary.
    -   `_conclude` Do some conclusion printing, cleaning up. Do not Calculate any results here use
                     `_calculate_results` mnethod instead.
    -   `_save_results` Save your results to a file. This is especially needed to use it from the command line.
2.  Choose an unique name and add `<analysis_example>` to the `__all__` list
    in `mdtools/ana/__init__.py`.
3.  OPTIONAL Add bash completion commands to `mdtools/share/mdtools_completion.bash`.
