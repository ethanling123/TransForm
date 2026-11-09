// protoclient fetches static payloads from the Caddy server over plain HTTP/1.1, HTTP/1.1 + TLS, HTTP/2 + TLS and
// HTTP/3, one fresh connection per measurement (cold) or a second request on a connection that has just carried one
// (warm), and prints one CSV row per measurement. Reported in REPORT.md section 19.
//
//	protoclient -host server -protos plain,h1,h2,h3 -files p100k.parquet,p1m.parquet -parallels 1,8 -modes cold -reps 5
//
// Bytes are the client NIC's receive and transmit counters across the measurement (headers, TLS, QUIC and ACKs
// included). Within a cell the order of the protocols is shuffled for every repetition.
package main

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/quic-go/quic-go/http3"
	"golang.org/x/net/http2"
)

type target struct {
	rt    http.RoundTripper
	base  string
	close func()
}

func newTarget(proto, host string) target {
	switch proto {
	case "plain":
		tr := &http.Transport{MaxIdleConnsPerHost: 64}
		return target{tr, "http://" + host + ":8080", tr.CloseIdleConnections}
	case "h1":
		tr := &http.Transport{
			TLSClientConfig:     &tls.Config{InsecureSkipVerify: true},
			TLSNextProto:        map[string]func(string, *tls.Conn) http.RoundTripper{},
			MaxIdleConnsPerHost: 64,
		}
		return target{tr, "https://" + host + ":8443", tr.CloseIdleConnections}
	case "h2":
		tr := &http2.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true, NextProtos: []string{"h2"}}}
		return target{tr, "https://" + host + ":8443", tr.CloseIdleConnections}
	case "h3":
		tr := &http3.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
		return target{tr, "https://" + host + ":8443", func() { tr.Close() }}
	}
	panic("unknown protocol " + proto)
}

type result struct {
	ttfb, done time.Duration
	n          int64
	err        error
}

func fetch(rt http.RoundTripper, url string, t0 time.Time) result {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	resp, err := rt.RoundTrip(req)
	if err != nil {
		return result{err: err}
	}
	ttfb := time.Since(t0)
	n, err := io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return result{ttfb, time.Since(t0), n, err}
}

func nic(name string) int64 {
	b, err := os.ReadFile("/sys/class/net/eth0/statistics/" + name)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	return v
}

func measure(proto, host, file string, parallel int, warm bool) (ttfb, firstDone, wall float64, bytes, rx, tx int64, err error) {
	tg := newTarget(proto, host)
	defer tg.close()
	url := tg.base + "/" + file
	if warm {
		if r := fetch(tg.rt, url, time.Now()); r.err != nil {
			return 0, 0, 0, 0, 0, 0, r.err
		}
		time.Sleep(300 * time.Millisecond)
	}
	rx0, tx0 := nic("rx_bytes"), nic("tx_bytes")
	t0 := time.Now()
	results := make([]result, parallel)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = fetch(tg.rt, url, t0)
		}(i)
	}
	wg.Wait()
	wall = time.Since(t0).Seconds()
	time.Sleep(300 * time.Millisecond)
	rx, tx = nic("rx_bytes")-rx0, nic("tx_bytes")-tx0
	ttfb, firstDone = 1e9, 1e9
	for _, r := range results {
		if r.err != nil {
			return 0, 0, 0, 0, 0, 0, r.err
		}
		bytes += r.n
		ttfb = min(ttfb, r.ttfb.Seconds())
		firstDone = min(firstDone, r.done.Seconds())
	}
	return
}

func main() {
	host := flag.String("host", "server", "server host")
	protos := flag.String("protos", "plain,h1,h2,h3", "protocols to compare")
	files := flag.String("files", "p100k.parquet,p1m.parquet,p8m.parquet", "payload files")
	parallels := flag.String("parallels", "1", "numbers of simultaneous requests")
	modes := flag.String("modes", "cold", "cold, warm (warm only runs with one request)")
	reps := flag.Int("reps", 5, "measured repetitions per cell (one extra warm-up repetition is dropped)")
	profile := flag.String("profile", "unshaped", "label for the link profile")
	flag.Parse()

	fmt.Println("profile,proto,file,parallel,mode,rep,ttfb_s,first_done_s,wall_s,bytes,rx,tx")
	protoList := strings.Split(*protos, ",")
	for _, file := range strings.Split(*files, ",") {
		for _, ps := range strings.Split(*parallels, ",") {
			parallel, _ := strconv.Atoi(ps)
			for _, mode := range strings.Split(*modes, ",") {
				if mode == "warm" && parallel != 1 {
					continue
				}
				for rep := 0; rep <= *reps; rep++ {
					order := append([]string(nil), protoList...)
					rand.New(rand.NewSource(int64(rep))).Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
					for _, proto := range order {
						ttfb, first, wall, bytes, rx, tx, err := measure(proto, *host, file, parallel, mode == "warm")
						if err != nil {
							fmt.Fprintf(os.Stderr, "%s %s %s x%d %s rep %d: %v\n", *profile, proto, file, parallel, mode, rep, err)
						} else if rep > 0 {
							fmt.Printf("%s,%s,%s,%d,%s,%d,%.4f,%.4f,%.4f,%d,%d,%d\n", *profile, proto, file, parallel, mode, rep, ttfb, first, wall, bytes, rx, tx)
						}
						time.Sleep(500 * time.Millisecond)
					}
				}
			}
		}
	}
}
