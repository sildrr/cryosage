"""Distinguish transient TLS record failures from certificate configuration errors."""


def certificate_failure(error):
    text = str(error).lower()
    return any(token in text for token in (
        "certificate_verify_failed", "certificate verify failed", "hostname mismatch",
        "self-signed certificate", "certificate has expired",
    ))


def transient_tls_failure(error):
    if certificate_failure(error):
        return False
    text = str(error).lower()
    return any(token in text for token in (
        "record layer failure", "unexpected eof while reading",
        "eof occurred in violation of protocol", "ssl_error_syscall",
    ))
