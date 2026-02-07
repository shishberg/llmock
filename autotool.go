package llmock

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// generateToolCallFromSchema picks a tool from the request and generates
// arguments conforming to its JSON schema. The rng is used for all random
// choices. If tools is empty, returns an empty ToolCall and false.
func generateToolCallFromSchema(tools []RequestTool, rng *rand.Rand) (ToolCall, bool) {
	if len(tools) == 0 {
		return ToolCall{}, false
	}

	tool := tools[rng.IntN(len(tools))]
	args := generateFromSchema(tool.Parameters, rng)

	argsMap, ok := args.(map[string]any)
	if !ok {
		argsMap = make(map[string]any)
	}

	return ToolCall{
		ID:        generateToolCallID("call_"),
		Name:      tool.Name,
		Arguments: argsMap,
	}, true
}

// generateFromSchema generates a value conforming to a JSON schema object.
// It handles type, properties, required, enum, items, and nested schemas.
func generateFromSchema(schema map[string]any, rng *rand.Rand) any {
	if schema == nil {
		return map[string]any{}
	}

	// Handle enum first — pick a random value regardless of type.
	if enum, ok := schema["enum"]; ok {
		if arr, ok := enum.([]any); ok && len(arr) > 0 {
			return arr[rng.IntN(len(arr))]
		}
	}

	typ, _ := schema["type"].(string)

	switch typ {
	case "object":
		return generateObject(schema, rng)
	case "array":
		return generateArray(schema, rng)
	case "string":
		return generateString(schema, rng)
	case "number":
		return generateNumber(rng)
	case "integer":
		return generateInteger(rng)
	case "boolean":
		return rng.IntN(2) == 0
	case "null":
		return nil
	default:
		// If type is unspecified but properties exist, treat as object.
		if _, ok := schema["properties"]; ok {
			return generateObject(schema, rng)
		}
		return map[string]any{}
	}
}

// generateObject creates a map with values for all required properties
// and optionally some non-required ones.
func generateObject(schema map[string]any, rng *rand.Rand) map[string]any {
	result := make(map[string]any)

	props, _ := schema["properties"].(map[string]any)
	if props == nil {
		return result
	}

	required := make(map[string]bool)
	if reqArr, ok := schema["required"].([]any); ok {
		for _, v := range reqArr {
			if s, ok := v.(string); ok {
				required[s] = true
			}
		}
	}

	for name, propSchema := range props {
		propMap, ok := propSchema.(map[string]any)
		if !ok {
			continue
		}
		// Always include required properties, include optional ones 50% of the time.
		if required[name] || rng.IntN(2) == 0 {
			result[name] = generateFromSchema(propMap, rng)
		}
	}

	return result
}

// generateArray creates a slice with 1-3 items matching the items schema.
func generateArray(schema map[string]any, rng *rand.Rand) []any {
	count := 1 + rng.IntN(3)
	itemSchema, _ := schema["items"].(map[string]any)

	items := make([]any, count)
	for i := range items {
		items[i] = generateFromSchema(itemSchema, rng)
	}
	return items
}

// sampleStrings are used when no enum or specific constraints are given.
var sampleStrings = []string{
	"hello", "world", "test", "example", "foo", "bar",
	"sample", "data", "value", "mock", "item", "entry",
}

// generateString returns a string value. If the schema name or format
// gives hints, it uses domain-appropriate values.
func generateString(schema map[string]any, rng *rand.Rand) string {
	if format, ok := schema["format"].(string); ok {
		switch format {
		case "date":
			return fmt.Sprintf("2024-%02d-%02d", 1+rng.IntN(12), 1+rng.IntN(28))
		case "date-time":
			return fmt.Sprintf("2024-%02d-%02dT%02d:%02d:%02dZ",
				1+rng.IntN(12), 1+rng.IntN(28), rng.IntN(24), rng.IntN(60), rng.IntN(60))
		case "email":
			return sampleStrings[rng.IntN(len(sampleStrings))] + "@example.com"
		case "uri", "url":
			return "https://example.com/" + sampleStrings[rng.IntN(len(sampleStrings))]
		}
	}

	// Use description to generate a more fitting value if possible.
	if desc, ok := schema["description"].(string); ok {
		lower := strings.ToLower(desc)
		switch {
		case strings.Contains(lower, "location") || strings.Contains(lower, "city"):
			cities := []string{"San Francisco", "New York", "London", "Tokyo", "Berlin", "Paris"}
			return cities[rng.IntN(len(cities))]
		case strings.Contains(lower, "name"):
			names := []string{"Alice", "Bob", "Charlie", "Dana", "Eve"}
			return names[rng.IntN(len(names))]
		case strings.Contains(lower, "language"):
			langs := []string{"en", "fr", "de", "ja", "es"}
			return langs[rng.IntN(len(langs))]
		}
	}

	return sampleStrings[rng.IntN(len(sampleStrings))]
}

func generateNumber(rng *rand.Rand) float64 {
	return float64(rng.IntN(1000)) / 10.0
}

func generateInteger(rng *rand.Rand) int {
	return rng.IntN(100)
}
