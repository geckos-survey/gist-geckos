import importlib.util
import logging
import os

from printStatus import printStatus

from gistPipeline.writeFITS import _writeFITS


def continuumCube_Module(config):
    """
    This function calls the continuum cube creation routine specified by the user.
    """
    printStatus.module("continuumCube module")

    # Check if module is turned off in MasterConfig
    if config["CONT"]["METHOD"] == False:
        message = "The module was turned off."
        printStatus.warning(message)
        logging.warning(message)
        return None

    # Check if outputs are already available
    outPrefix = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
    if (
        config["GENERAL"]["OW_OUTPUT"] == False
        and os.path.isfile(outPrefix + "_CONTcube.fits") == True
        and os.path.isfile(outPrefix + "_ORIGcube.fits") == True
        and os.path.isfile(outPrefix + "_LINEcube.fits") == True
    ):
        logging.info(
            "Results of the module are already in the output directory. Module is skipped."
        )
        printStatus.done("Results are already available. Module is skipped.")
        return None

    # Import the chosen continuumCube routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config["CONT"]["METHOD"]
            + "_cont_wrapper.py",
        )
        logging.info(
            "Using the continuumCube routine '" + config["CONT"]["METHOD"] + "_cont_wrapper.py'"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the continuumCube routine '"
            + config["CONT"]["METHOD"]
            + "_cont_wrapper.py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen continuumCube routine
    try:
        module.createContinuumCube(config)
        _writeFITS.generateFITS(config, "CONT")
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "continuumCube routine '" + config["CONT"]["METHOD"] + "_cont_wrapper.py' failed."
        )
        printStatus.failed(message + " See LOGFILE for further information.")
        logging.critical(message)
        return "SKIP"

    # Return
    return None
