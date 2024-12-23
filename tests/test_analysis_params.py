from mapmanagercore.analysis_params import AnalysisParams
from mapmanagercore.logger import logger

def test_analysis_params():
    ap = AnalysisParams()
    for k,v in ap.getDict().items():
        if k == "__version__":
            continue
        # if 'type' not in v.keys():
        #     logger.error('')
        requiredKeys = ['defaultValue', 'currentValue', 'description', 'type']
        for requiredKey in requiredKeys:
            assert requiredKey in v.keys(), f'did not find required key "{requiredKey}".'

    ap.resetDefaults()
