#[cfg(test)]
mod tests {
    #[test]
    fn test_calculate_nfft() {
        assert_eq!(calculate_nfft(4,4), 16);
    }

    #[test]
    fn test_mel2hz() {
        assert_eq!(mel2hz(150), 99.6513453291);
    }

    #[test]
    fn test_hz2mel() {
        assert_eq!(hz2mel(100), 150.491278896);
    }
}
