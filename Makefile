.PHONY: build crawl index search dashboard test clean

build:
	docker-compose build

crawl:
	docker-compose run --rm crawler

index:
	docker-compose run --rm indexer

search:
	docker-compose up search dashboard

test:
	python -m pytest tests/ -v

clean:
	docker-compose down
	docker system prune -f