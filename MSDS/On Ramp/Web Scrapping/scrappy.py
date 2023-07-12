# Version 4
# PythonDocumentationSpider
# Using XPath to extract all information
# Using DocSectionItem to return data
from scrapy.crawler import CrawlerProcess
import scrapy

class DocSectionItem(scrapy.Item):
    # section_name attribute is of type scrapy.Field()
    section_name = scrapy.Field()
    section_link = scrapy.Field()

class PythonDocumentationSpider(scrapy.Spider):
    name = 'pydoc_bot'
    start_urls = ['https://docs.python.org/3/']
    custom_settings = {
        'ITEM_PIPELINES': {
            '__main__.ExtractFirstLine': 1
        },
        'FEEDS': {
            'quotes.csv': {
                'format': 'csv',
                'overwrite': True
            }
        }
    }
    def parse(self, response):
        for a_el in response.xpath('//table[@class="contentstable"]//a[@class="biglink"]'):
            section = DocSectionItem()
            section['section_name'] = a_el.xpath('./text()').extract()[0]
            section['section_link'] = a_el.xpath('./@href').extract()[0]
            print(type(section))
            yield section
# process = CrawlerProcess()
# process.crawl(PythonDocumentationSpider)
# process.start()