.PHONY: init up down logs restart clean status

init:
	@[ -f .env ] || cp .env.example .env
	@echo "Generated .env (edit secrets if going beyond local dev)."
	@echo "Then run: make up"

up:
	docker compose up -d
	@echo ""
	@echo "langfuse UI: http://localhost:3000"
	@echo "minio UI:    http://localhost:9091   (login minio/miniosecret)"
	@echo ""
	@echo "First time: sign up in UI → create org → create project → copy keys to Sorting-Hat/.env.local"

down:
	docker compose down

restart:
	docker compose restart langfuse-web langfuse-worker

logs:
	docker compose logs -f --tail=100 langfuse-web langfuse-worker

status:
	docker compose ps

clean:
	docker compose down -v
	@echo "All Langfuse data wiped."
