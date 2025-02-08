from pprint import pprint
from mapmanagercore.lazy_geo_pd_images.metadata import Metadata

# test v1 of Metadata

def test_meta_data():
    md = Metadata()

    pprint(md)

    # add color channel

if __name__ == '__main__':
    test_meta_data()