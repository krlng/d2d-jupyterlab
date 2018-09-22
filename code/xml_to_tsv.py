import sys
import os
import re
import xml.etree.ElementTree

try:  # python2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    pass


def usage(msg):
    if msg:
        sys.stderr.write('{}\n'.format(msg))
    sys.stderr.write('python xml_to_tsv.py input output\n\n')
    sys.stderr.write('\tinput \t input data set to convert to TSV.\n')
    sys.stderr.write('\touput \t file to write an output.\n')
    sys.exit(1)


if len(sys.argv) != 3:
    usage('Wrong number of arguments. Usage:')


input = sys.argv[1]
output = sys.argv[2]
tag = 'python'


def process_posts(fd_in, fd_out, target_tag):
    num = 1
    for line in fd_in:
        try:
            attr = xml.etree.ElementTree.fromstring(line).attrib

            pid = attr.get('Id', '')
            label = 1 if target_tag in attr.get('Tags', '') else 0
            title = re.sub('\s+', ' ', attr.get('Title', '')).strip()
            body = re.sub('\s+', ' ', attr.get('Body', '')).strip()
            text = title + ' ' + body

            fd_out.write(u'{}\t{}\t{}\n'.format(pid, label, text))

            num += 1
        except Exception as ex:
            sys.stderr.write('Warning! Error in line {}: {}\n'.format(num, ex))


target_tag = u'<' + tag + '>'


with open(input) as fd_in:
    with open(output, 'w') as fd_out:
        process_posts(fd_in, fd_out, target_tag)

