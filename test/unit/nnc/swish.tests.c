#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _test_swish_mul_autodiff(const int wrt, int* const __case_result__)
{
	const float beta = 1.7;
	const float scale = 0.25;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t value = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 7), "value");
	const ccv_nnc_tensor_symbol_t gate = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 7), "gate");
	const ccv_nnc_tensor_symbol_t output = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "output");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_MUL_FORWARD(beta, scale), TENSOR_SYMBOL_LIST(value, gate), TENSOR_SYMBOL_LIST(output), "swish_mul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	if (wrt == 3)
		ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(output), TENSOR_SYMBOL_LIST(value, gate), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	else if (wrt == 1)
		ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(output), TENSOR_SYMBOL_LIST(value), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	else
		ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(output), TENSOR_SYMBOL_LIST(gate), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t doutput = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, output);
	const ccv_nnc_tensor_symbol_t dvalue = (wrt & 1) ? ccv_nnc_tensor_symbol_for_backward(symbolic_graph, value) : NO_TENSOR_SYMBOL;
	const ccv_nnc_tensor_symbol_t dgate = (wrt & 2) ? ccv_nnc_tensor_symbol_for_backward(symbolic_graph, gate) : NO_TENSOR_SYMBOL;
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	if (wrt == 3)
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, TENSOR_SYMBOL_LIST(output, dvalue, dgate), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	else if (wrt == 1)
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, TENSOR_SYMBOL_LIST(output, dvalue), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	else
		ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, TENSOR_SYMBOL_LIST(output, dgate), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* const value_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, value);
	ccv_nnc_tensor_t* const gate_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, gate);
	ccv_nnc_tensor_t* const doutput_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, doutput);
	int i;
	for (i = 0; i < 7; i++)
	{
		value_tensor->data.f32[i] = (float)(i - 3) * 0.4f;
		gate_tensor->data.f32[i] = (float)(i - 2) * 0.35f;
		doutput_tensor->data.f32[i] = (float)(i + 1) * 0.125f;
	}
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dvalue_tensor = (wrt & 1) ? ccv_nnc_tensor_from_symbol(tensor_arena, dvalue) : 0;
	ccv_nnc_tensor_t* const dgate_tensor = (wrt & 2) ? ccv_nnc_tensor_from_symbol(tensor_arena, dgate) : 0;
	const float epsilon = 1e-3;
	for (i = 0; i < 7; i++)
	{
		const float v = value_tensor->data.f32[i];
		const float g = gate_tensor->data.f32[i];
		const float incoming_gradient = doutput_tensor->data.f32[i];
		if (dvalue_tensor)
		{
			const float plus = scale * (v + epsilon) * g / (1 + expf(-beta * g));
			const float minus = scale * (v - epsilon) * g / (1 + expf(-beta * g));
			const float numerical = incoming_gradient * (plus - minus) / (2 * epsilon);
			REQUIRE_EQ_WITH_TOLERANCE(numerical, dvalue_tensor->data.f32[i], 1e-4, "swish mul autodiff value gradient should match numerical gradient");
		}
		if (dgate_tensor)
		{
			const float plus_gate = g + epsilon;
			const float minus_gate = g - epsilon;
			const float plus = scale * v * plus_gate / (1 + expf(-beta * plus_gate));
			const float minus = scale * v * minus_gate / (1 + expf(-beta * minus_gate));
			const float numerical = incoming_gradient * (plus - minus) / (2 * epsilon);
			REQUIRE_EQ_WITH_TOLERANCE(numerical, dgate_tensor->data.f32[i], 1e-4, "swish mul autodiff gate gradient should match numerical gradient");
		}
	}
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish mul autodiff uses canonical backward inputs")
{
	_test_swish_mul_autodiff(3, __case_result__);
	_test_swish_mul_autodiff(1, __case_result__);
	_test_swish_mul_autodiff(2, __case_result__);
}

TEST_CASE("compare swish gradient with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	const ccv_nnc_tensor_symbol_t sigmoid = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "sigmoid");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(sigmoid), "sigmoid");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(x, sigmoid), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(y, dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const swish_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t sx = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	ccv_nnc_tensor_symbol_t sy = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(swish_symbolic_graph, CMD_SWISH_FORWARD(1), TENSOR_SYMBOL_LIST(sx), TENSOR_SYMBOL_LIST(sy), "swish");
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(swish_symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(swish_symbolic_graph, TENSOR_SYMBOL_LIST(sy), TENSOR_SYMBOL_LIST(sx), SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t sdx = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sx);
	const ccv_nnc_tensor_symbol_t sdy = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sy);
	ccv_nnc_graph_t* swish_graph = 0;
	ccv_nnc_tensor_arena_t* swish_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* swish_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(swish_symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(sy, sdx),
		SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph),
		&swish_graph, &swish_tensor_arena, &swish_graph_exec_arena);
	ccv_nnc_tensor_t* const sx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sx);
	memcpy(sx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 10);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const sdy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdy);
	for (i = 0; i < 10; i++)
		sdy_tensor->data.f32[i] = dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(swish_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const sy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sy);
	REQUIRE_TENSOR_EQ(y_tensor, sy_tensor, "graph computed result should match swish op result");
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const sdx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdx);
	REQUIRE_TENSOR_EQ(dx_tensor, sdx_tensor, "gradient computed result should match swish op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(swish_symbolic_graph);
	ccv_nnc_tensor_arena_free(swish_tensor_arena);
	ccv_nnc_graph_exec_arena_free(swish_graph_exec_arena);
	ccv_nnc_graph_free(swish_graph);
}

TEST_CASE("swish with non-one beta")
{
	const float beta = 1.75;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32), "x");
	const ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "bx");
	const ccv_nnc_tensor_symbol_t sigmoid = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "sigmoid");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(beta), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "beta_x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(sigmoid), "sigmoid");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(x, sigmoid), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(y, dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 42);
	for (i = 0; i < 32; i++)
		x_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 4 - 2);
	ccv_nnc_symbolic_graph_t* const swish_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t sx = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, CPU_TENSOR_NHWC(32F, 32), "x");
	ccv_nnc_tensor_symbol_t sy = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(swish_symbolic_graph, CMD_SWISH_FORWARD(beta), TENSOR_SYMBOL_LIST(sx), TENSOR_SYMBOL_LIST(sy), "swish");
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(swish_symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(swish_symbolic_graph, TENSOR_SYMBOL_LIST(sy), TENSOR_SYMBOL_LIST(sx), SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t sdx = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sx);
	const ccv_nnc_tensor_symbol_t sdy = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sy);
	ccv_nnc_graph_t* swish_graph = 0;
	ccv_nnc_tensor_arena_t* swish_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* swish_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(swish_symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(sy, sdx),
		SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph),
		&swish_graph, &swish_tensor_arena, &swish_graph_exec_arena);
	ccv_nnc_tensor_t* const sx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sx);
	memcpy(sx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 32);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const sdy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdy);
	for (i = 0; i < 32; i++)
		sdy_tensor->data.f32[i] = dy_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(swish_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const sy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sy);
	REQUIRE_TENSOR_EQ(y_tensor, sy_tensor, "graph computed result should match swish op result");
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const sdx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdx);
	REQUIRE_TENSOR_EQ(dx_tensor, sdx_tensor, "gradient computed result should match swish op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(swish_symbolic_graph);
	ccv_nnc_tensor_arena_free(swish_tensor_arena);
	ccv_nnc_graph_exec_arena_free(swish_graph_exec_arena);
	ccv_nnc_graph_free(swish_graph);
}

TEST_CASE("swish gradient with non-one beta")
{
	const float beta = 0.5;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 64), "x");
	const ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "bx");
	const ccv_nnc_tensor_symbol_t sigmoid = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "sigmoid");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(beta), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "beta_x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(sigmoid), "sigmoid");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(x, sigmoid), TENSOR_SYMBOL_LIST(y), "y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 97);
	for (i = 0; i < 64; i++)
		x_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 6 - 3);
	ccv_nnc_symbolic_graph_t* const swish_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t sx = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, CPU_TENSOR_NHWC(32F, 64), "x");
	ccv_nnc_tensor_symbol_t sy = ccv_nnc_tensor_symbol_new(swish_symbolic_graph, ccv_nnc_tensor_auto, "y");
	ccv_nnc_graph_exec_symbol_new(swish_symbolic_graph, CMD_SWISH_FORWARD(beta), TENSOR_SYMBOL_LIST(sx), TENSOR_SYMBOL_LIST(sy), "swish");
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(swish_symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_backward(swish_symbolic_graph, TENSOR_SYMBOL_LIST(sy), TENSOR_SYMBOL_LIST(sx), SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(swish_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t sdx = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sx);
	const ccv_nnc_tensor_symbol_t sdy = ccv_nnc_tensor_symbol_for_backward(swish_symbolic_graph, sy);
	ccv_nnc_graph_t* swish_graph = 0;
	ccv_nnc_tensor_arena_t* swish_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* swish_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(swish_symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(sdx),
		SYMBOLIC_GRAPH_SOURCES(swish_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(swish_symbolic_graph),
		&swish_graph, &swish_tensor_arena, &swish_graph_exec_arena);
	ccv_nnc_tensor_t* const sx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sx);
	memcpy(sx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 64);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	ccv_nnc_tensor_t* const sdy_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdy);
	for (i = 0; i < 64; i++)
		sdy_tensor->data.f32[i] = dy_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(swish_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const sdx_tensor = ccv_nnc_tensor_from_symbol(swish_tensor_arena, sdx);
	REQUIRE_TENSOR_EQ(dx_tensor, sdx_tensor, "gradient computed result should match swish op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(swish_symbolic_graph);
	ccv_nnc_tensor_arena_free(swish_tensor_arena);
	ccv_nnc_graph_exec_arena_free(swish_graph_exec_arena);
	ccv_nnc_graph_free(swish_graph);
}

#include "case_main.h"
