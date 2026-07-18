// Package main is the custom Endpoint Picker (EPP) for RFC-0001: the stock
// llm-d-router runner plus this repo's out-of-tree plugins registered before startup.
// Everything else (flags, config-file parsing, in-tree plugin registration) is the
// upstream runner's.
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

	classaware "gateway-plugin"
)

func main() {
	os.Exit(run())
}

func run() int {
	ctx := ctrl.SetupSignalHandler()

	fwkplugin.Register(classaware.KVCachePriorityType, classaware.KVCachePriorityFactory)
	fwkplugin.Register(classaware.ClassAwareReliabilityType, classaware.ClassAwareReliabilityFactory)

	if err := runner.NewRunner().Run(ctx); err != nil {
		return 1
	}
	return 0
}
