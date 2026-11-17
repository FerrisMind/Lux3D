//! Outbound media URL policy for `source_url` / `mesh_url` fetches.
//!
//! The server fetches user-supplied URLs, which makes it an SSRF target. This
//! module applies the policy shared by the media source resolver:
//!
//! * only `http` and `https` URLs, without embedded credentials;
//! * every resolved address must be public (loopback, RFC1918/ULA, link-local,
//!   unspecified, multicast, broadcast, CGNAT and IPv4-mapped IPv6 addresses
//!   are rejected);
//! * DNS rebinding cannot bypass the check: [`MediaUrlResolver`] validates the
//!   exact addresses returned to the HTTP connector;
//! * redirects are revalidated per hop (at most [`MEDIA_FETCH_MAX_REDIRECTS`]),
//!   including the scheme downgrade and private-address rules;
//! * IP literals are checked before the request because the HTTP connector
//!   resolves them without consulting the DNS resolver.
//!
//! `allow_private_urls` (server `--allow-private-urls` /
//! `LUX3D_ALLOW_PRIVATE_URLS=1`) disables only the non-public address filter,
//! for LAN deployments. Scheme, credentials, redirect and size/timeout rules
//! keep applying. Documentation-only ranges (for example 192.0.2.0/24) are
//! deliberately not classified as non-public; they are not internal targets.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{Context, Result, bail};
use reqwest::{
    dns::{Addrs, Name, Resolve, Resolving},
    redirect::Policy,
};
use url::{Host, Url};

use crate::config::MEDIA_FETCH_MAX_REDIRECTS;

/// The actionable suffix appended to non-public address errors.
const PRIVATE_URL_HINT: &str = "pass `--allow-private-urls` (server) or set `LUX3D_ALLOW_PRIVATE_URLS=1` to allow LAN/private hosts";

/// Returns `true` when an outbound media fetch must not connect to `ip`.
///
/// Rejected IPv4: unspecified/`0.0.0.0/8`, loopback, RFC1918, link-local,
/// multicast, broadcast and CGNAT (`100.64.0.0/10`). Rejected IPv6:
/// unspecified, loopback, link-local (`fe80::/10`), unique-local (`fc00::/7`),
/// multicast and every IPv4-mapped address (`::ffff:a.b.c.d`); mapped
/// addresses are rejected outright so an IPv4 target cannot bypass the IPv4
/// classifier through its IPv6 spelling.
pub fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_disallowed_ipv4(ip),
        IpAddr::V6(ip) => {
            if ip.to_ipv4_mapped().is_some() {
                return true;
            }
            let segments = ip.segments();
            // Deprecated IPv4-compatible `::/96` spellings (`::a.b.c.d`) can
            // reach IPv4 targets on stacks that still route them; reject the
            // whole range (unspecified/loopback are already covered).
            if segments[..6].iter().all(|segment| *segment == 0) {
                return true;
            }
            ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_multicast()
                || (segments[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (segments[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || (segments[0] & 0xffc0) == 0xfec0 // fec0::/10 deprecated site-local
        }
    }
}

fn is_disallowed_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip == Ipv4Addr::BROADCAST
        || octets[0] == 0 // 0.0.0.0/8 "this network"
        || (octets[0] == 100 && (octets[1] & 0xc0) == 64) // 100.64.0.0/10 CGNAT
}

/// Parse and validate a media URL before the first request.
///
/// Returns the parsed URL so callers do not re-parse (and cannot accidentally
/// validate one spelling and fetch another).
pub fn validate_media_url(url: &str, allow_private_urls: bool) -> Result<Url> {
    let parsed = Url::parse(url).with_context(|| format!("invalid media URL `{url}`"))?;
    validate_scheme(&parsed)?;
    reject_credentials(&parsed)?;
    if parsed.host_str().is_none() {
        bail!("media URL `{parsed}` has no host");
    }
    if !allow_private_urls {
        reject_non_public_literal(&parsed)?;
    }
    Ok(parsed)
}

/// Revalidate a redirect before following it.
///
/// `hops` is the number of requests already made in the redirect chain
/// (the initial request counts as one, matching reqwest's
/// `Attempt::previous().len()`).
pub fn validate_redirect_target(
    previous: Option<&Url>,
    next: &Url,
    hops: usize,
    allow_private_urls: bool,
) -> Result<()> {
    if hops > MEDIA_FETCH_MAX_REDIRECTS {
        bail!("too many redirects (max {MEDIA_FETCH_MAX_REDIRECTS})");
    }
    validate_scheme(next)?;
    if previous.is_some_and(|previous| previous.scheme() == "https" && next.scheme() == "http") {
        bail!("redirect from `https` to `http` is not allowed");
    }
    reject_credentials(next)?;
    if next.host_str().is_none() {
        bail!("redirect target `{next}` has no host");
    }
    if !allow_private_urls {
        reject_non_public_literal(next)?;
    }
    Ok(())
}

fn validate_scheme(url: &Url) -> Result<()> {
    match url.scheme() {
        "http" | "https" => Ok(()),
        scheme => bail!(
            "media URL scheme `{scheme}` is not allowed in `{url}`; only `http` and `https` are supported"
        ),
    }
}

fn reject_credentials(url: &Url) -> Result<()> {
    if !url.username().is_empty() || url.password().is_some() {
        bail!("media URL `{url}` must not include credentials");
    }
    Ok(())
}

fn reject_non_public_literal(url: &Url) -> Result<()> {
    let Some(ip) = literal_ip(url) else {
        return Ok(());
    };
    if is_disallowed_ip(ip) {
        bail!("media URL host `{ip}` is not a public address; {PRIVATE_URL_HINT}");
    }
    Ok(())
}

fn literal_ip(url: &Url) -> Option<IpAddr> {
    match url.host()? {
        Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
        Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
        Host::Domain(_) => None,
    }
}

/// Resolver used by the media HTTP client: resolves with the system resolver
/// and rejects the whole answer when any address is non-public.
///
/// Validating the exact address list returned to the connector closes the
/// DNS-rebinding race: a second lookup returning a private address is never
/// used because no second lookup happens.
#[derive(Debug, Clone, Copy)]
pub struct MediaUrlResolver {
    allow_private_urls: bool,
}

impl MediaUrlResolver {
    pub fn new(allow_private_urls: bool) -> Self {
        Self { allow_private_urls }
    }
}

impl Resolve for MediaUrlResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_owned();
        let allow_private_urls = self.allow_private_urls;
        Box::pin(async move {
            // Port 0 is deliberate: reqwest replaces it with the URL port.
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|error| format!("failed to resolve media URL host `{host}`: {error}"))?
                .collect();
            if addrs.is_empty() {
                return Err(format!("media URL host `{host}` resolved to no addresses").into());
            }
            if !allow_private_urls {
                for addr in &addrs {
                    if is_disallowed_ip(addr.ip()) {
                        return Err(format!(
                            "media URL host `{host}` resolves to non-public address `{}`; {PRIVATE_URL_HINT}",
                            addr.ip()
                        )
                        .into());
                    }
                }
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

/// Redirect policy used by the media HTTP client.
///
/// The policy enforces the redirect count, scheme and credential rules and the
/// literal-IP half of the address filter on every hop. Hostnames are covered
/// by [`MediaUrlResolver`] when the redirect target is connected.
pub fn media_redirect_policy(allow_private_urls: bool) -> Policy {
    Policy::custom(move |attempt| {
        match validate_redirect_target(
            attempt.previous().last(),
            attempt.url(),
            attempt.previous().len(),
            allow_private_urls,
        ) {
            Ok(()) => attempt.follow(),
            Err(error) => attempt.error(error),
        }
    })
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    use reqwest::dns::Resolve;
    use url::Url;

    use super::{MediaUrlResolver, is_disallowed_ip, validate_media_url, validate_redirect_target};
    use crate::config::MEDIA_FETCH_MAX_REDIRECTS;

    fn ip(text: &str) -> IpAddr {
        text.parse().expect("valid IP")
    }

    #[test]
    fn classifier_rejects_non_public_ipv4() {
        for text in [
            "0.0.0.0",
            "0.1.2.3",
            "127.0.0.1",
            "127.255.255.254",
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.127.255.255",
            "224.0.0.1",
            "239.255.255.255",
            "255.255.255.255",
        ] {
            assert!(
                is_disallowed_ip(ip(text)),
                "`{text}` must be rejected as non-public"
            );
        }
    }

    #[test]
    fn classifier_allows_public_ipv4() {
        for text in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "172.32.0.1",
            "100.128.0.1",
        ] {
            assert!(
                !is_disallowed_ip(ip(text)),
                "`{text}` must be allowed as public"
            );
        }
    }

    #[test]
    fn classifier_rejects_non_public_ipv6_and_mapped_addresses() {
        for text in [
            "::",
            "::1",
            "fe80::1",
            "febf:ffff::1",
            "fc00::1",
            "fd12:3456::1",
            "fec0::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
            "::ffff:169.254.169.254",
            // IPv4-mapped addresses are rejected even when the mapped IPv4
            // address is public, so the IPv4 classifier cannot be bypassed.
            "::ffff:8.8.8.8",
            // Deprecated IPv4-compatible spellings (`::/96`).
            "::7f00:1",
            "::808:808",
        ] {
            assert!(
                is_disallowed_ip(ip(text)),
                "`{text}` must be rejected as non-public"
            );
        }
    }

    #[test]
    fn classifier_allows_public_ipv6() {
        for text in [
            "2606:4700:4700::1111",
            "2001:4860:4860::8888",
            "2a00:1450:4001::1",
        ] {
            assert!(
                !is_disallowed_ip(ip(text)),
                "`{text}` must be allowed as public"
            );
        }
    }

    #[test]
    fn media_urls_allow_only_http_and_https_without_credentials() {
        validate_media_url("https://example.com/image.png", false).expect("https URL");
        validate_media_url("http://example.com/image.png", false).expect("http URL");

        for url in [
            "ftp://example.com/image.png",
            "file:///etc/passwd",
            "data:text/plain,hello",
        ] {
            let error = validate_media_url(url, false).expect_err("non-http scheme must fail");
            assert!(
                error.to_string().contains("scheme"),
                "unexpected message for `{url}`: `{error}`"
            );
        }
        let error = validate_media_url("https://user:secret@example.com/x.png", false)
            .expect_err("credentials must be rejected");
        assert!(
            error.to_string().contains("credentials"),
            "unexpected message: `{error}`"
        );
    }

    #[test]
    fn media_urls_reject_private_literals_unless_opted_in() {
        for url in [
            "http://127.0.0.1/x.png",
            "http://10.1.2.3/x.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/x.png",
            "http://[fe80::1]/x.png",
            "http://[::ffff:127.0.0.1]/x.png",
        ] {
            let error =
                validate_media_url(url, false).expect_err("private literal must be rejected");
            assert!(
                error.to_string().contains("not a public address"),
                "unexpected message for `{url}`: `{error}`"
            );
            validate_media_url(url, true).expect("the opt-in disables the address filter");
        }

        validate_media_url("http://93.184.216.34/x.png", false).expect("public literal");
    }

    #[test]
    fn redirect_policy_enforces_count_scheme_and_ips() {
        let previous = Url::parse("https://example.com/a").expect("URL");
        let https = Url::parse("https://example.com/b").expect("URL");
        let http = Url::parse("http://example.com/b").expect("URL");
        let ftp = Url::parse("ftp://example.com/b").expect("URL");
        let private = Url::parse("https://127.0.0.1/b").expect("URL");

        validate_redirect_target(Some(&previous), &https, 1, false).expect("https -> https");
        validate_redirect_target(Some(&previous), &http, 1, false)
            .expect_err("https -> http downgrade must fail");
        validate_redirect_target(Some(&http), &https, 1, false)
            .expect("http -> https upgrade is allowed");
        validate_redirect_target(Some(&previous), &ftp, 1, false)
            .expect_err("non-http redirect target must fail");
        validate_redirect_target(Some(&previous), &private, 1, false)
            .expect_err("private literal redirect target must fail without the opt-in");
        validate_redirect_target(Some(&previous), &private, 1, true)
            .expect("the opt-in disables the address filter for redirects");

        validate_redirect_target(Some(&previous), &https, MEDIA_FETCH_MAX_REDIRECTS, false)
            .expect("the last allowed hop");
        let error = validate_redirect_target(
            Some(&previous),
            &https,
            MEDIA_FETCH_MAX_REDIRECTS + 1,
            false,
        )
        .expect_err("one hop past the limit must fail");
        assert!(
            error.to_string().contains("too many redirects"),
            "unexpected message: `{error}`"
        );
    }

    #[tokio::test]
    async fn resolver_rejects_private_addresses_and_allows_them_on_opt_in() {
        fn expect_resolve_error(
            result: Result<super::Addrs, Box<dyn std::error::Error + Send + Sync>>,
        ) -> String {
            match result {
                Ok(_) => panic!("private resolution must fail"),
                Err(error) => error.to_string(),
            }
        }

        let name = "127.0.0.1".parse().expect("resolver name");
        let error = expect_resolve_error(MediaUrlResolver::new(false).resolve(name).await);
        assert!(
            error.contains("non-public address"),
            "unexpected message: `{error}`"
        );

        let name = "127.0.0.1".parse().expect("resolver name");
        let addrs: Vec<SocketAddr> = MediaUrlResolver::new(true)
            .resolve(name)
            .await
            .expect("opt-in resolution")
            .collect();
        assert_eq!(vec![SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)], addrs);

        // Numeric IPv6 hosts take the same path.
        let name = "::1".parse().expect("resolver name");
        expect_resolve_error(MediaUrlResolver::new(false).resolve(name).await);

        // A public numeric address is not filtered.
        let name = "93.184.216.34".parse().expect("resolver name");
        let addrs: Vec<SocketAddr> = MediaUrlResolver::new(false)
            .resolve(name)
            .await
            .expect("public resolution")
            .collect();
        assert_eq!(
            vec![SocketAddr::new(Ipv4Addr::new(93, 184, 216, 34).into(), 0)],
            addrs
        );
    }
}
