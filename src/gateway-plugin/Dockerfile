# Custom EPP image for RFC-0001 phase 4: the llm-d-router runner plus this repo's
# out-of-tree plugins (kv-cache-priority PreRequest, class-aware-reliability
# ProfileHandler). Adapted from llm-d-router's Dockerfile.epp; same runtime layout and
# ports so it drops into the GIE inferencepool chart via image override.
ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

FROM --platform=${BUILDPLATFORM} golang:1.26 AS go-builder

ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace

COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./
COPY cmd/ cmd/

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -ldflags="-s -w" -o bin/epp ./cmd/epp

FROM ${BASE_IMAGE}

WORKDIR /

COPY --from=go-builder /workspace/bin/epp /app/epp

USER 65532:65532

# gRPC ext-proc, health, metrics, KV-events ZMQ SUB — same as upstream Dockerfile.epp.
EXPOSE 9002
EXPOSE 9003
EXPOSE 9090
EXPOSE 5557

ENTRYPOINT ["/app/epp"]
