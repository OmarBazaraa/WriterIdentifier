import xml.etree.ElementTree as ET


def get_writer_id(filename):
    """
    Given a file names, find te file's writer id.
    :return: writer_id
    """
    tree = ET.parse("../data/raw/xml/" + filename + ".xml")
    root = tree.getroot()
    writer_id = int(root.attrib["writer-id"])
    return writer_id
