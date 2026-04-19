import { render, screen, fireEvent } from "@testing-library/react";
import Renderer from "../Renderer";
import type { WidgetContext } from "../Renderer";
import type {
  NumberFieldSpec, CheckBoxSpec, VerticalLayoutSpec,
} from "../../../api/types";

function makeCtx(initial: Record<string, unknown> = {}) {
  const state: Record<string, unknown> = { ...initial };
  const ctx: WidgetContext = {
    nodeId: "n1",
    propValue: (prop) => state[prop],
    onChange: (prop, value) => { state[prop] = value; },
  };
  return { ctx, state };
}

describe("Renderer", () => {
  it("NumberField renders current value and updates on change", () => {
    const { ctx, state } = makeCtx({ sigma: 1.0 });
    const spec: NumberFieldSpec = {
      kind: "NumberField", prop: "sigma", label: "Sigma",
      min: 0, max: 20, step: 0.1, decimals: 1, default: 1.0,
    };
    render(<Renderer spec={spec} ctx={ctx} />);
    const input = screen.getByLabelText("Sigma") as HTMLInputElement;
    expect(input.value).toBe("1");
    fireEvent.change(input, { target: { value: "2.5" } });
    expect(state.sigma).toBe(2.5);
  });

  it("CheckBox toggles value", () => {
    const { ctx, state } = makeCtx({ flag: false });
    const spec: CheckBoxSpec = {
      kind: "CheckBox", prop: "flag", label: "Flag", default: false,
    };
    render(<Renderer spec={spec} ctx={ctx} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(state.flag).toBe(true);
  });

  it("VerticalLayout renders children", () => {
    const { ctx } = makeCtx({ a: 1, b: false });
    const spec: VerticalLayoutSpec = {
      kind: "VerticalLayout",
      children: [
        { kind: "NumberField", prop: "a", label: "A",
          min: 0, max: 10, step: 1, decimals: 0, default: 0 },
        { kind: "CheckBox", prop: "b", label: "B", default: false },
      ],
    };
    render(<Renderer spec={spec} ctx={ctx} />);
    expect(screen.getByLabelText("A")).toBeInTheDocument();
    expect(screen.getByLabelText("B")).toBeInTheDocument();
  });
});
