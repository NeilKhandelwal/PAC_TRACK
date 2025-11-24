import React, { useEffect, useState } from 'react';

interface PACData {
    pac_name?: string;
    amount?: number;
    // add other fields as needed
}

interface Props {
    candidateId: string;
}

export default function CandidatePACInfo({ candidateId }: Props) {
    const [data, setData] = useState<PACData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchPAC() {
            try {
                const res = await fetch(`/api/pac/${candidateId}`);
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || 'Failed to fetch');
                }
                const json = await res.json();
                setData(json);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }
        if (candidateId) {
            fetchPAC();
        }
    }, [candidateId]);

    if (loading) return <p className="text-sm text-gray-500">Loading PAC info…</p>;
    if (error) return <p className="text-sm text-red-500">{error}</p>;
    if (!data) return null;

    return (
        <div className="mt-4 p-4 border rounded bg-gray-50 dark:bg-gray-800">
            <h2 className="text-lg font-semibold mb-2">PAC Funding</h2>
            <p className="text-sm">
                <strong>PAC:</strong> {data.pac_name ?? 'Unknown'}
            </p>
            {data.amount !== undefined && (
                <p className="text-sm">
                    <strong>Amount:</strong> ${data.amount.toLocaleString()}
                </p>
            )}
        </div>
    );
}
