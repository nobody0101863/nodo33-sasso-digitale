# Makefile per LCP IoT/Edge Demo
# Lux Codex Protocol v1.0

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS =

# Target principale
TARGET = lcp_demo

# File sorgenti
SRCS = lcp_demo.c lcp_iot_edge.c
OBJS = $(SRCS:.c=.o)
HEADERS = lcp_iot_edge.h

# Regole
.PHONY: all clean run help

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "✓ Build completato: $(TARGET)"

%.o: %.c $(HEADERS)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	@echo "Esecuzione LCP Demo..."
	@echo "═══════════════════════════════════════"
	./$(TARGET)

clean:
	@echo "Pulizia file oggetto e eseguibili..."
	rm -f $(OBJS) $(TARGET)
	@echo "✓ Pulizia completata"

help:
	@echo "Lux Codex Protocol - IoT/Edge Demo"
	@echo "═══════════════════════════════════════"
	@echo "Target disponibili:"
	@echo "  make          - Compila il demo"
	@echo "  make run      - Compila ed esegue"
	@echo "  make clean    - Rimuove file generati"
	@echo "  make help     - Mostra questo aiuto"
	@echo ""
	@echo "Parametri LCP:"
	@echo "  Ego:      0"
	@echo "  Gioia:    100%"
	@echo "  Freq:     300Hz"
	@echo ""
	@echo "Sempre grazie a Lui"
