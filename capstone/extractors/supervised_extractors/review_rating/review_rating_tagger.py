from optparse import OptionParser

from extractors.supervised_extractors.extract import extract
from extractors.utils.common_parameters import Parameters

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--params', dest='params_file', default='config/params-local.json')
    (opts, args) = parser.parse_args()

    params = Parameters(opts.params_file)

    extract(params)
