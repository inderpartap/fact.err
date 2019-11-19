import scrapy


class collectingSites(scrapy.Spider):
    name="news_scraper"

    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'test.csv'
    }
    start_urls = ['http://worldnewsdailyreport.com/new-childhood-pictures-of-justin-trudeau-come-back-to-haunt-him']

    def parse(self,response):
        SET_SELECTOR = '.entry-content'

        for headlineDiv in response.css(SET_SELECTOR):

            Headline_Selector = 'h2 ::text'
            text_selector ='.//p/text()'
            textList =  response.xpath(text_selector).getall()
            textAll = " ".join(textList)
            yield{
                'headline':headlineDiv.css(Headline_Selector).extract_first(),
                'description':textAll
            }

            next_pages_selector = '.related-content'
            next_pages = response.css(next_pages_selector)
            for page in next_pages.xpath('.//article'):
                next_link = page.xpath('.//div/a/@href').get()
                yield scrapy.Request(
                    url=next_link,
                    callback =self.parse
                )







