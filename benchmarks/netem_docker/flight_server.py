"""Arrow Flight server for the transport baseline: the same tables and batch size as the TransForm stream arms.

A ticket (or a flight descriptor's command) is JSON: {"table": "tpcds" | "mixed", "codec": "none" | "lz4" | "zstd"}.
do_get streams RecordBatches of ROWS_PER_CHUNK rows. The "stats" action returns the number of batches pulled from the
generator since the last "reset", which is how the client sees whether the server kept producing after a cancel.
"""

from __future__ import annotations

import json
import threading

import pyarrow as pa
import pyarrow.flight as flight

from benchmarks.netem_docker.transport_common import FLIGHT_PORT, ROWS_PER_CHUNK, tables


class TransportFlightServer(flight.FlightServerBase):
    def __init__(self, location: str):
        super().__init__(location)
        self._tables = tables()
        self._lock = threading.Lock()
        self._batches_pulled = 0

    def _spec(self, raw: bytes) -> tuple[pa.Table, str]:
        spec = json.loads(raw)
        return self._tables[spec["table"]], spec["codec"]

    def get_flight_info(self, context, descriptor):
        table, _ = self._spec(descriptor.command)
        endpoint = flight.FlightEndpoint(flight.Ticket(descriptor.command), [])
        return flight.FlightInfo(table.schema, descriptor, [endpoint], table.num_rows, -1)

    def do_get(self, context, ticket):
        table, codec = self._spec(ticket.ticket)

        def batches():
            for batch in table.to_batches(max_chunksize=ROWS_PER_CHUNK):
                with self._lock:
                    self._batches_pulled += 1
                yield batch

        options = pa.ipc.IpcWriteOptions(compression=None if codec == "none" else codec)
        return flight.GeneratorStream(table.schema, batches(), options=options)

    def do_action(self, context, action):
        if action.type == "reset":
            with self._lock:
                self._batches_pulled = 0
            yield flight.Result(b"{}")
        elif action.type == "stats":
            with self._lock:
                yield flight.Result(json.dumps({"batches_pulled": self._batches_pulled}).encode())
        else:
            raise flight.FlightServerError(f"unknown action {action.type}")


if __name__ == "__main__":
    server = TransportFlightServer(f"grpc+tcp://0.0.0.0:{FLIGHT_PORT}")
    print(f"flight server on {FLIGHT_PORT}", flush=True)
    server.serve()
