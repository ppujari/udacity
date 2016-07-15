import codecs
import os
from string import Template

from pytagcloud import create_html_data


def generate_tag_cloud(tags, output_file):
    data = create_html_data(tags, size=(900, 600), fontname='Crimson Text', layout=0)

    template_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/template.html'), 'r')
    html_template = Template(template_file.read())

    context = {}

    tags_template = '''
        <li class="cnt" style="top: %(top)dpx; left: %(left)dpx; height: %(height)dpx;">
            <a class="tag %(cls)s" href="#%(tag)s" style="top: %(top)dpx; left: %(left)dpx; font-size: %(size)dpx; height: %(height)dpx; line-height:%(lh)dpx;">%(tag)s</a>
        </li>
    '''

    context['tags'] = ''.join([tags_template.rstrip() % link for link in data['links']])
    context['width'] = data['size'][0]
    context['height'] = data['size'][1]
    context['css'] = '\n'.join("a.%(cname)s{color:%(normal)s;}\
    a.%(cname)s:hover{color:%(hover)s;}" %
                              {'cname':k,
                               'normal': v[0],
                               'hover': v[1]}
                             for k,v in data['css'].items())


    html_text = html_template.substitute(context)

    with codecs.open('cloud.html', 'w', encoding='utf-8') as html_file:
        html_file.write(html_text)

