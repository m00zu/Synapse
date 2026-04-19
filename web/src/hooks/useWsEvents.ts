import { useEffect } from "react";
import { api } from "../api/client";
import type { WsEvent } from "../api/types";
import { useGraph } from "../store/graph";

export function useWsEvents() {
  const applyEvent = useGraph((s) => s.applyWsEvent);
  useEffect(() => {
    const ws = api.openWs(applyEvent);
    return () => { ws.close(); };
  }, [applyEvent]);
}

// Optional overload for tests: manual dispatch.
export function dispatchWsEvent(ev: WsEvent) {
  useGraph.getState().applyWsEvent(ev);
}
