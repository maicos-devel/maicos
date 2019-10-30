# Developing for maicos

Contribution via pull requests are always welcome. 
Before submitting a pull request please open an issue to discuss your 
changes. 

The main feauture branch is 
the `develop` branch. Use this for submitting your requests. The `master` branch 
contains  all commits of the latest release. 
More information on the branching model we used is given in this 
[nice blog post](https://nvie.com/posts/a-successful-git-branching-model/).

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
    in `maicos/modules/__init__.py`.
3.  OPTIONAL Add bash completion commands to `maicos/share/maicos_completion.bash`.

# UnitTests

The tests also rely on the `pytest` library and use some work flows
 from `numpy` and `MDAnalysisTests`. In order to run the tests you need those packages.

To start the test process just simply type from the root of the repository

    cd test
    pytest  --disable-pytest-warnings

Whenever you add a new feature to the code you should also add a test case.
Furthermore test cases are also useful if a bug is fixed or anything you
 think worthwhile. Follow the philosophy - the more the better!